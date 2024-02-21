import yaml

compose_template = r"""
services:
  {MODEL_VERSION}:
    build: Dockerfile.{MODEL_VERSION}
    ports: 
      - "{HOST_PORT}:{FLASK_PORT}"
"""

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    compose_ins = compose_template.replace(
        "{MODEL_VERSION}", str(cfg["model_version"])
        ).replace(
        "{HOST_PORT}", str(cfg["host_port"])
        ).replace(
        "{FLASK_PORT}", str(cfg["flask_port"])
        )
    with open("docker-compose.yaml", "w") as f:
        f.write(compose_ins)

