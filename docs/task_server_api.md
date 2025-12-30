# Task Server API Specification

This document defines the standard HTTP API that task servers must implement to be compatible with AReaL's `TaskServerWorkflow`.

## Overview

Task servers provide interactive environments for RL training. They manage episode lifecycle, execute actions, and compute rewards. AReaL handles the RL training, model inference, and trajectory collection.

## Base URL

All endpoints are relative to the task server's base URL (e.g., `http://localhost:5000/api`).

## Endpoints

### 1. Get Task Information

```
GET /task/info
```

Returns metadata about the task.

**Response:**
```json
{
  "name": "os-interaction",
  "num_samples": 1000,
  "max_episode_length": 8,
  "observation_type": "text",
  "action_type": "text",
  "description": "Operating system interaction tasks"
}
```

### 2. Start Episode

```
POST /episode/start
```

Initializes a new episode for a specific task sample.

**Request:**
```json
{
  "sample_id": "dev-001-00042",
  "config": {
    "seed": 42
  }
}
```

**Response:**
```json
{
  "episode_id": "uuid-1234-5678",
  "observation": {
    "type": "text",
    "content": "You are an assistant that will act like a person..."
  },
  "info": {
    "max_turns": 8,
    "task_description": "Tell me how many files are in /etc",
    "sample_id": "dev-001-00042"
  }
}
```

### 3. Step Episode

```
POST /episode/step
```

Executes an action in the environment and returns the next observation.

**Request:**
```json
{
  "episode_id": "uuid-1234-5678",
  "action": {
    "type": "text",
    "content": "Think: I need to count files.\nAct: bash\n```bash\nls /etc | wc -l\n```"
  }
}
```

**Response (continuing):**
```json
{
  "episode_id": "uuid-1234-5678",
  "observation": {
    "type": "text",
    "content": "The output of the OS:\n220"
  },
  "reward": 0.0,
  "done": false,
  "info": {
    "turn": 1,
    "parsed_action": {
      "thought": "I need to count files.",
      "action": "bash",
      "content": "ls /etc | wc -l"
    }
  }
}
```

**Response (terminal):**
```json
{
  "episode_id": "uuid-1234-5678",
  "observation": null,
  "reward": 1.0,
  "done": true,
  "info": {
    "success": true,
    "num_turns": 3,
    "answer": "220",
    "status": "completed"
  }
}
```

### 4. Cancel Episode

```
POST /episode/cancel
```

Cancels a running episode and cleans up resources.

**Request:**
```json
{
  "episode_id": "uuid-1234-5678"
}
```

**Response:**
```json
{
  "status": "cancelled",
  "episode_id": "uuid-1234-5678"
}
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Episode or sample not found
- `500 Internal Server Error`: Server-side error

Error responses include details:

```json
{
  "error": "Episode not found",
  "episode_id": "uuid-1234-5678",
  "detail": "Episode may have expired or been cancelled"
}
```

## State Management

- Each episode maintains isolated state (e.g., Docker container for OS tasks)
- Episodes automatically timeout after 300 seconds of inactivity
- Resources are cleaned up when episodes complete or are cancelled

## Observation Types

Currently supported observation types:

- `text`: Plain text observations (most common)
- `json`: Structured JSON observations (future)
- `image`: Base64-encoded images (future, for visual tasks)

## Action Types

Currently supported action types:

- `text`: Free-form text actions (most common)
- `structured`: JSON-formatted actions (future)
