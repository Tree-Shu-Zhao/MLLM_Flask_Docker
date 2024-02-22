import yaml
import shutil

compose_template = r"""
services:
  {MODEL_VERSION}:
    build:
      context: .
      dockerfile: Dockerfile
    ports: 
      - "{HOST_PORT}:{FLASK_PORT}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['{GPU_ID}']
              capabilities: [gpu]
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
        ).replace(
        "{GPU_ID}", str(cfg["gpu_id"])
        )
    with open("docker-compose.yaml", "w") as f:
        f.write(compose_ins)

    shutil.copy(f"./dockerfiles/Dockerfile.{cfg['model_version']}", "./Dockerfile")

