// Command layerdiff compares two model checkpoints tensor by tensor:
// streamed hashes, numeric stats, and a changed-layer report.
package main

import (
	"os"

	"github.com/JaydenCJ/layerdiff/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
