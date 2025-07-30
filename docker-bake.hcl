# Docker Bake configuration for Master Thesis project
# Usage: docker buildx bake --push

variable "REGISTRY" {
  default = "lukashinterleitner"
}

variable "IMAGE_NAME" {
  default = "master-thesis-data-science"
}

# Git commit hash for versioning (matches GitHub Actions ${{ github.sha }})
variable "IMAGE_TAG" {
  default = ""

  validation {
    condition = IMAGE_TAG != ""
    error_message = "IMAGE_TAG must not be empty"
  }

  validation {
    condition = can(regex("^[0-9a-f]{7,40}$", IMAGE_TAG))
    error_message = "IMAGE_TAG must be a valid Git commit hash (7-40 hex characters)"
  }
}

# Default group - builds main target
group "default" {
  targets = ["app"]
}

# Main application target - matches GitHub Actions workflow
target "app" {
  dockerfile = "Dockerfile"
  context = "."
  tags = [
    "${REGISTRY}/${IMAGE_NAME}:${substr(IMAGE_TAG, 0, 8)}",
    "${REGISTRY}/${IMAGE_NAME}:latest"
  ]
  labels = {
    "org.opencontainers.image.title" = "Master Thesis - LLM Training Data Explanations"
    "org.opencontainers.image.description" = "Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations"
    "org.opencontainers.image.authors" = "lukashinterleitner"
    "org.opencontainers.image.source" = "https://github.com/lukas-hinterleitner/master-thesis"
    "org.opencontainers.image.revision" = "${IMAGE_TAG}"
  }
  output = ["type=registry"]
}

# Local development target (no push to registry)
target "local" {
  dockerfile = "Dockerfile"
  context = "."
  tags = [
    "${IMAGE_NAME}:local"
  ]
  platforms = ["linux/amd64"]
  output = ["type=docker"]
}
