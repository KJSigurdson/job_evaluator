"""Supabase client construction. Service-role key bypasses RLS — server-side use only."""
from __future__ import annotations

import os


def get_client():
    """Build a Supabase client from SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY."""
    from supabase import Client, create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)
