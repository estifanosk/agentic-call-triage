/*
 * Case-management service: the system of record the agent writes into.
 *
 * This is the shape most enterprises actually land on. The agent layer is Python
 * because that is where the model tooling lives; the system of record stays on the JVM
 * because that is where the transactional services, the auth model, and the operational
 * maturity already are. They meet over an HTTP contract, and neither side needs to know
 * how the other is built.
 *
 * The service is deliberately not a rubber stamp. It re-validates every case on arrival:
 * an approved-by-a-human flag from an upstream agent is a claim, not a guarantee, and a
 * system of record that trusts its callers has no integrity story. Server-side validation
 * is the last line where "the model produced something odd" stops being a data problem.
 *
 * Single-file, JDK built-ins only. Run with:  java java/CaseService.java
 */

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CaseService {

    private static final int PORT = Integer.parseInt(
            System.getenv().getOrDefault("CASE_SERVICE_PORT", "8080"));

    private static final Set<String> VALID_SEVERITIES = Set.of("none", "low", "medium", "high");

    private static final Map<String, Map<String, String>> CASES = new ConcurrentHashMap<>();
    private static final AtomicInteger SEQUENCE = new AtomicInteger(1000);

    public static void main(String[] args) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);

        // Virtual threads: this workload is almost entirely IO wait, so a thread per
        // request costs next to nothing on Loom and the code stays synchronous.
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());

        server.createContext("/cases", CaseService::handleCases);
        server.createContext("/health", exchange ->
                respond(exchange, 200, "{\"status\":\"up\"}"));

        server.start();
        System.out.printf("case-management service listening on http://localhost:%d%n", PORT);
        System.out.println("  POST /cases      file an approved case");
        System.out.println("  GET  /cases      list filed cases");
        System.out.println("  GET  /health     liveness");
    }

    private static void handleCases(HttpExchange exchange) throws IOException {
        try {
            switch (exchange.getRequestMethod()) {
                case "POST" -> createCase(exchange);
                case "GET" -> listCases(exchange);
                default -> respond(exchange, 405, error("method not allowed"));
            }
        } catch (RuntimeException e) {
            respond(exchange, 500, error("internal error: " + e.getClass().getSimpleName()));
        }
    }

    private static void createCase(HttpExchange exchange) throws IOException {
        String body;
        try (InputStream in = exchange.getRequestBody()) {
            body = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }

        String callId = extract(body, "call_id");
        String severity = extract(body, "severity");
        String headline = extract(body, "headline");

        List<String> problems = new ArrayList<>();
        if (callId.isBlank()) {
            problems.add("call_id is required");
        }
        if (!VALID_SEVERITIES.contains(severity)) {
            problems.add("severity must be one of " + VALID_SEVERITIES);
        }
        if (headline.isBlank()) {
            problems.add("headline is required");
        }
        if (headline.length() > 200) {
            problems.add("headline exceeds 200 characters");
        }

        if (!problems.isEmpty()) {
            System.out.printf("[reject] call=%s %s%n", callId, problems);
            respond(exchange, 422, error(String.join("; ", problems)));
            return;
        }

        // Idempotency on call_id. Agents retry - on a timeout, a transient tool error, or
        // a replayed checkpoint - and a case-management system that creates a duplicate
        // complaint every time is worse than one that occasionally drops a write.
        Map<String, String> existing = CASES.get(callId);
        if (existing != null) {
            System.out.printf("[dedupe] call=%s -> existing %s%n", callId, existing.get("case_id"));
            respond(exchange, 200, String.format(
                    "{\"status\":\"duplicate\",\"case_id\":\"%s\",\"filed_at\":\"%s\"}",
                    existing.get("case_id"), existing.get("filed_at")));
            return;
        }

        String caseId = "CASE-" + SEQUENCE.incrementAndGet();
        String filedAt = Instant.now().toString();
        CASES.put(callId, Map.of(
                "case_id", caseId,
                "call_id", callId,
                "severity", severity,
                "headline", headline,
                "filed_at", filedAt));

        System.out.printf("[filed] %s call=%s severity=%s%n", caseId, callId, severity);
        respond(exchange, 201, String.format(
                "{\"status\":\"created\",\"case_id\":\"%s\",\"filed_at\":\"%s\"}", caseId, filedAt));
    }

    private static void listCases(HttpExchange exchange) throws IOException {
        StringBuilder out = new StringBuilder("{\"cases\":[");
        boolean first = true;
        for (Map<String, String> record : CASES.values()) {
            if (!first) {
                out.append(',');
            }
            first = false;
            out.append(String.format(
                    "{\"case_id\":\"%s\",\"call_id\":\"%s\",\"severity\":\"%s\",\"headline\":\"%s\"}",
                    record.get("case_id"), record.get("call_id"),
                    record.get("severity"), escape(record.get("headline"))));
        }
        out.append("]}");
        respond(exchange, 200, out.toString());
    }

    /*
     * Minimal string-field reader. A real service uses Jackson; hand-rolling it here
     * keeps the file dependency-free and runnable with the single-file source launcher.
     */
    private static String extract(String json, String field) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(field) + "\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
                .matcher(json);
        return m.find() ? m.group(1).replace("\\\"", "\"").replace("\\n", " ") : "";
    }

    private static String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String error(String message) {
        return String.format("{\"status\":\"error\",\"detail\":\"%s\"}", escape(message));
    }

    private static void respond(HttpExchange exchange, int status, String body) throws IOException {
        byte[] payload = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, payload.length);
        try (var out = exchange.getResponseBody()) {
            out.write(payload);
        }
    }
}
