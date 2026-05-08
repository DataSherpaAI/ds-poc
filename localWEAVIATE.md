# Weaviate Docker Deployment on a VM

A hardened workflow for running a single-node Weaviate instance inside a Docker container on a Linux VM, intended to be consumed **only** by an application running on the same VM. The application is the only thing exposed to the internet; Weaviate itself is bound to `127.0.0.1` and protected by an API key.

## Assumptions

- Target OS: **Ubuntu 24.04 LTS**.
- Recommended minimum specs for a small RAG workload: **2 vCPU, 4 GB RAM, 20 GB disk**. Production workloads scale roughly with vector count and dimensionality — size accordingly.
- You have `sudo` privileges.
- The application that will consume Weaviate runs on the same VM (or in another container on the same Docker network) and connects via `127.0.0.1` (or the Compose service name, if containerized alongside).

---

## A. VM readiness

### A1. Verify OS and resources

```bash
lsb_release -a
uname -a
df -h /            # confirm root partition has space
df -h /var/lib     # this is where Docker volumes live by default
free -h
nproc
```

You want to see free disk on whichever partition hosts `/var/lib/docker`, since persistent Weaviate data will land in a Docker volume there.

### A2. Add swap (optional but recommended)

Swap acts as a safety net so the OOM killer does not terminate Weaviate (or other services) during transient memory spikes — common during bulk imports or HNSW index builds. Set swappiness low so the kernel only uses swap under real pressure.

The block below is **idempotent** — safe to re-run.

```bash
# Create swapfile only if it does not already exist
if [ ! -f /swapfile ]; then
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
fi

# Add to fstab only if not already present
grep -q '^/swapfile ' /etc/fstab || \
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Set swappiness (drop-in file is overwritten cleanly on re-run)
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swappiness.conf
sudo sysctl --system

# Verify
free -h
swapon --show
```

**Sizing rule of thumb:** swap = RAM, capped at 8 GB for most VMs. Tune up for very memory-hungry workloads.

---

## B. Install Docker Engine + Compose plugin

Follows Docker's official Ubuntu instructions.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the Docker apt repo
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

sudo systemctl enable --now docker

# Verify
docker --version
docker compose version
sudo docker run --rm hello-world
```

### B1. Run Docker without sudo (recommended)

```bash
sudo usermod -aG docker $USER
```

> **Important:** group membership only takes effect in a new login session. Either log out and back in, or run `newgrp docker` for the current shell. After that, verify:

```bash
docker ps
```

---

## C. Create the deployment directory

```bash
sudo mkdir -p /opt/weaviate
sudo chown -R "$USER":"$USER" /opt/weaviate
cd /opt/weaviate
```

---

## D. Generate credentials

Generate a strong API key and write it directly to the `.env` file (avoids clipboard exposure and copy/paste mistakes):

```bash
cd /opt/weaviate
umask 077

API_KEY="$(openssl rand -hex 32)"
cat > /opt/weaviate/.env <<EOF
WEAVIATE_API_KEYS=${API_KEY}
WEAVIATE_API_USERS=rag-client
WEAVIATE_ROOT_USERS=rag-client
EOF

chmod 600 /opt/weaviate/.env

# Confirm permissions
ls -l /opt/weaviate/.env
```

**Save the key somewhere safe** (password manager) — your application will need it as `WEAVIATE_API_KEY`. To retrieve it later from the VM:

```bash
grep '^WEAVIATE_API_KEYS=' /opt/weaviate/.env | cut -d= -f2-
```

> **Note on multiple keys/users:** `WEAVIATE_API_KEYS` and `WEAVIATE_API_USERS` are comma-separated lists with **matching arity** — the Nth key belongs to the Nth user. If you add a second client later, both lists must grow together.

---

## E. `docker-compose.yml`

Create `/opt/weaviate/docker-compose.yml`:

```yaml
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.36.8
    container_name: weaviate
    restart: unless-stopped
    command:
      - --host
      - 0.0.0.0
      - --port
      - "8080"
      - --scheme
      - http

    # SECURITY: bind to localhost only — not reachable from outside the VM.
    # The application running on the same VM connects via 127.0.0.1.
    ports:
      - "127.0.0.1:8080:8080"   # REST
      - "127.0.0.1:50051:50051" # gRPC

    volumes:
      - weaviate_data:/var/lib/weaviate

    environment:
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
      CLUSTER_HOSTNAME: "node1"

      # RAG default: vectors are computed by the application; Weaviate stores them.
      DEFAULT_VECTORIZER_MODULE: "none"
      ENABLE_MODULES: ""

      QUERY_DEFAULTS_LIMIT: "25"

      # Auth: API keys only (no anonymous access)
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "false"
      AUTHENTICATION_APIKEY_ENABLED: "true"
      AUTHENTICATION_APIKEY_ALLOWED_KEYS: "${WEAVIATE_API_KEYS}"
      AUTHENTICATION_APIKEY_USERS: "${WEAVIATE_API_USERS}"

      # RBAC
      AUTHORIZATION_RBAC_ENABLED: "true"
      AUTHORIZATION_RBAC_ROOT_USERS: "${WEAVIATE_ROOT_USERS}"

      # Memory hint for Go's GC. Set to ~90% of the container's memory limit
      # (or available RAM if no limit). Adjust to match deploy.resources.limits below.
      GOMEMLIMIT: "3GiB"

    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:8080/v1/.well-known/ready || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 30s

    # Resource ceiling — prevents Weaviate from starving other VM processes.
    # Tune to your VM size. Compose v2 honors these on standalone Docker.
    deploy:
      resources:
        limits:
          memory: 3500M

volumes:
  weaviate_data:
```

**Notes on choices:**

- **Image tag `1.36.8`** is the current GA version in Weaviate's official docs at the time of writing. Pinning to a specific patch version is the right call for reproducibility — bump it intentionally during planned upgrades.
- **`container_name: weaviate`** lets you run `docker logs weaviate` directly. Without it, Compose generates `weaviate-weaviate-1`.
- **Healthcheck** uses Weaviate's `/v1/.well-known/ready` endpoint, which is intentionally unauthenticated.
- **Resource limits and `GOMEMLIMIT`** should be adjusted for your VM size. The pair above suits a 4 GB VM; on an 8 GB VM, try `7000M` and `GOMEMLIMIT: "6GiB"`.
- **gRPC port 50051** is included because newer Weaviate clients prefer gRPC for performance. Remove that line if you only use REST.

---

## F. Start Weaviate and verify

```bash
cd /opt/weaviate
docker compose pull
docker compose up -d
docker compose ps
```

Watch the logs until you see Weaviate report it is serving:

```bash
docker compose logs -f weaviate
# Ctrl-C to stop tailing once you see "Serving weaviate at..." or similar
```

After ~30 seconds the healthcheck should mark the container `healthy`:

```bash
docker compose ps
```

### Verification checks

**1. Liveness/readiness (no auth required, must succeed):**

```bash
curl -fsS http://127.0.0.1:8080/v1/.well-known/ready && echo OK
```

**2. Anonymous access is disabled (must return 401):**

```bash
curl -i http://127.0.0.1:8080/v1/schema | head -n 1
# Expected: HTTP/1.1 401 Unauthorized
```

**3. Authenticated access works (must return 200):**

```bash
# Pull the key from the .env file rather than retyping it
set -a; source /opt/weaviate/.env; set +a
API_KEY="${WEAVIATE_API_KEYS}"

curl -fsS -H "Authorization: Bearer ${API_KEY}" \
  http://127.0.0.1:8080/v1/schema | head
```

**4. Confirm the port is not reachable from outside the VM:**

From your laptop or another host:

```bash
curl --max-time 5 http://<VM_PUBLIC_IP>:8080/v1/.well-known/ready
# Expected: connection refused or timeout
```

---

## Wiring the application to Weaviate

The application running on the same VM (or in another container on the same Compose network) should be configured with:

```
WEAVIATE_URL=http://127.0.0.1:8080
WEAVIATE_API_KEY=<contents of WEAVIATE_API_KEYS from /opt/weaviate/.env>
```

If you later move the application into the same `docker-compose.yml`, switch the URL to `http://weaviate:8080` and you can drop the host port publish for REST entirely (keep gRPC if needed).

---

## Operational cheatsheet

```bash
# Status
docker compose -f /opt/weaviate/docker-compose.yml ps
docker compose -f /opt/weaviate/docker-compose.yml logs --tail=200 weaviate

# Restart
docker compose -f /opt/weaviate/docker-compose.yml restart weaviate

# Stop / start
docker compose -f /opt/weaviate/docker-compose.yml down
docker compose -f /opt/weaviate/docker-compose.yml up -d

# Upgrade Weaviate (after editing the image tag)
docker compose -f /opt/weaviate/docker-compose.yml pull
docker compose -f /opt/weaviate/docker-compose.yml up -d

# Backup (cold backup — stop first for a consistent snapshot)
docker compose -f /opt/weaviate/docker-compose.yml down
sudo tar -czf /opt/weaviate/backup-$(date +%F).tgz \
  -C /var/lib/docker/volumes/weaviate_weaviate_data/_data .
docker compose -f /opt/weaviate/docker-compose.yml up -d
```

For warm/online backups, use Weaviate's built-in [Backup module](https://docs.weaviate.io/weaviate/configuration/backups) with a filesystem or S3 backend — preferable in production.

---

## Optional hardening

- **Firewall.** Even though Weaviate is bound to `127.0.0.1`, a default-deny `ufw` policy that only allows the application's public ports (e.g., 80/443) is good hygiene. Note that Docker's iptables rules can bypass `ufw` for published ports — but since nothing here is published on `0.0.0.0`, this is mostly belt-and-suspenders.
- **Automatic security updates.** `sudo apt install unattended-upgrades` and enable for security pocket only.
- **Log rotation.** Docker's default JSON log driver can grow indefinitely; set `log-opts` (e.g., `max-size: "50m"`, `max-file: "3"`) in `/etc/docker/daemon.json`.
- **Monitoring.** Consider adding `cAdvisor` or Prometheus's `node_exporter` if you need visibility into resource usage over time.
