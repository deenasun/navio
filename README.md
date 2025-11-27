# Navio

## Setup
Create a new virtual environment

1. Ensure you have Python 3.11+ installed: `python3 --version`
2. Create the venv: `python3 -m venv .venv`
3. Activate it:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
4. Install project dependencies: `pip install -r requirements.txt`
5. Add `SUPABASE_URL` and `SUPABASE_KEY` for the Navio Supabase project in `.env`
6. Add an API key to access Google AI Studio in `.env`

## MCP Server

1. Start the MCP server in developer mode:
```bash
mcp dev server.py
```

2. Navigate to the MCP Inspector URL in your browser.

3. Make sure transport type is set to `STDIO` and click on "Connect"

4. List tools/make tool calls etc. to make sure the MCP server is working as expected

## Connecting the MCP Server to MCP Clients
1. Make a copy of `mcp_template.json` and replace the `command` and `cwd` fields.

2. Save `mcp_template.json` as `mcp.json` in your IDE's preferred MCP configuration location.

3. You should be able to prompt an agent to use the MCP tools inside `server.py`