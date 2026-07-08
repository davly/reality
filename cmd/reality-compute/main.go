package main

import (
	"log"
	"net/http"
	"os"
	"time"
)

// Environment configuration. Both have safe defaults: an unset service token
// makes the producer fail-closed (every /mcp/tools call -> 401), and the port
// defaults to 8090.
const (
	// envServiceToken is the machine-trust shared secret. MUST equal Nexus's
	// FLAGSHIP_TOOL_PROVIDERS token field (and the REALITY_SERVICE_TOKEN a
	// Shape-2 provider would forward). UNSET => fail-closed (401 for all).
	envServiceToken = "NEXUS_SERVICE_TOKEN"
	// envPort overrides the listen port.
	envPort = "PORT"

	defaultPort = "8090"
)

func main() {
	serviceToken := os.Getenv(envServiceToken)
	if serviceToken == "" {
		// Fail-closed by design: we still start (so the deployment is reachable
		// and observable), but every /mcp/tools call returns 401 until the
		// operator provisions the secret. Loud once, at startup.
		log.Printf("reality-compute: WARNING %s is unset — running FAIL-CLOSED; all /mcp/tools calls will 401 until it is set", envServiceToken)
	}

	srv := buildServer(serviceToken, os.Getenv(envPort))

	log.Printf("reality-compute: listening on %s (capability: %s)", srv.Addr, toolConformalInterval)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("reality-compute: server error: %v", err)
	}
}

// buildServer assembles the configured *http.Server from the resolved service
// token and (possibly empty) port. Factored out of main so the wiring — port
// defaulting, timeout budget, and the fail-closed mux — is unit-testable
// without binding a socket.
func buildServer(serviceToken, port string) *http.Server {
	if port == "" {
		port = defaultPort
	}
	return &http.Server{
		Addr:    ":" + port,
		Handler: newMux(serviceToken),
		// Conformal compute is pure CPU and fast; generous-but-bounded timeouts
		// protect against slow-loris and runaway clients (gosec G114).
		ReadTimeout:       10 * time.Second,
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20, // 1 MiB
	}
}
