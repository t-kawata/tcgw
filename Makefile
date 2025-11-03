run-bifrost:
	@echo "RUN Bifrost"
	./bifrost/bifrost-darwin-arm64-v1.3.13 -app-dir ./bifrost -host 0.0.0.0 -port 7766
run-main:
	@echo "RUN main.go"
	cd ./src && go run main.go
run-all:
	@echo "Starting all servers..."
	./bifrost/bifrost-darwin-arm64-v1.3.13 -app-dir ./bifrost -host 0.0.0.0 -port 7766 & sleep 3 && cd ./src && go run main.go
