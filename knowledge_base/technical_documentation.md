# Technical Documentation

## API Guidelines

Our REST API follows standard HTTP conventions. All endpoints require authentication via Bearer tokens.

### Rate Limiting
- 1000 requests per hour for authenticated users
- 100 requests per hour for unauthenticated users

### Error Handling
All errors return JSON with the following structure:
```json
{
  "error": "error_code",
  "message": "Human readable error message",
  "details": {}
}
```

## Database Schema

The user table contains the following fields:
- id (primary key)
- email (unique)
- password_hash
- created_at
- updated_at