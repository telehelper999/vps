# app.py (integrated)
# Uses Telethon (Telegram USER) + FastAPI backend + Redis pub/sub (Node.js handles Socket.IO)
# Incorporates the user's code extraction idea (regex) where possible.
import os, re, asyncio, random, string
from typing import List, Dict, Any, Optional, Set
from fastapi import FastAPI, Query, Depends, HTTPException, Body, Request, Form
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import hashlib
import secrets
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError
from keepalive import KeepAliveService
# ⬇️ add:
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime, func, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
# ⬇️ Redis Integration for secure pub/sub:
from redis_config import RedisManager

PORT = int(os.getenv("PORT", "5000"))
TG_API_ID = int(os.getenv("TG_API_ID", "0") or "0")
TG_API_HASH = os.getenv("TG_API_HASH", "")
TG_SESSION = os.getenv("TG_SESSION", "tg_session")  # file path or session string
CHANNELS = os.getenv("CHANNELS", "-1002772030545,-1002932455889,-1001992047801")  # Multiple channels separated by comma

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0"))
WS_SECRET = os.getenv("WS_SECRET")

# Authentication Configuration
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "secure123")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))

# ⬇️ Enhanced Session Storage with Redis:
redis_manager = None
try:
    redis_manager = RedisManager()
    if redis_manager.test_connection():
        print("✅ Redis connected securely with password authentication")
        # Use Redis for session storage in production
        active_sessions = set()  # Fallback for Redis unavailable
    else:
        print("⚠️ Redis not available - using in-memory sessions")
        active_sessions = set()
except Exception as e:
    print(f"⚠️ Redis initialization failed: {e} - using in-memory sessions")
    active_sessions = set()

# Enhanced regex patterns for different code formats
CODE_PATTERNS = [
    r'(?i)Code:\s+([a-zA-Z0-9]{4,25})',           # "Code: stakecomrtlye4" - primary pattern
    r'(?i)Code:([a-zA-Z0-9]{4,25})',              # "Code:stakecomguft19f6" - no space version
    r'(?i)Bonus:\s+([a-zA-Z0-9]{4,25})',         # "Bonus: ABC123"
    r'(?i)Bonus:([a-zA-Z0-9]{4,25})',            # "Bonus:ABC123" 
    r'(?i)Claim:\s+([a-zA-Z0-9]{4,25})',         # "Claim: ABC123"
    r'(?i)Claim:([a-zA-Z0-9]{4,25})',            # "Claim:ABC123"
    r'(?i)Promo:\s+([a-zA-Z0-9]{4,25})',         # "Promo: ABC123"
    r'(?i)Promo:([a-zA-Z0-9]{4,25})',            # "Promo:ABC123"
    r'(?i)Coupon:\s+([a-zA-Z0-9]{4,25})',        # "Coupon: ABC123"
    r'(?i)Coupon:([a-zA-Z0-9]{4,25})',           # "Coupon:ABC123"
    r'(?i)use\s+(?:code\s+)?([a-zA-Z0-9]{4,25})',  # "use code ABC123"
    r'(?i)enter\s+(?:code\s+)?([a-zA-Z0-9]{4,25})', # "enter code ABC123"
]

# Pattern for extracting both code and value from messages like:
# Code: stakecomlop1n84b
# Value: $3
CODE_VALUE_PATTERN = r'(?i)Code:\s+([a-zA-Z0-9]{4,25})(?:.*?\n.*?Value:\s+\$?(\d+(?:\.\d{1,2})?))?'

CLAIM_URL_BASE = os.getenv("CLAIM_URL_BASE", "https://autoclaim.example.com")
RING_SIZE = int(os.getenv("RING_SIZE", "100"))

# Keep-alive configuration
RENDER_URL = os.getenv("RENDER_URL", "https://tester-px62.onrender.com")
KEEP_ALIVE_ENABLED = os.getenv("KEEP_ALIVE_ENABLED", "true").lower() == "true"
KEEP_ALIVE_INTERVAL = int(os.getenv("KEEP_ALIVE_INTERVAL", "5"))  # minutes

# ⬇️ Enhanced Database Connection with Auto-Management:
DATABASE_URL = os.getenv("DATABASE_URL")
Base = declarative_base()

# Robust database initialization
engine = None
if DATABASE_URL:
    try:
        # Enhanced engine with connection pooling and auto-disconnect
        engine = create_engine(
            DATABASE_URL, 
            pool_pre_ping=True,           # Test connections before use
            pool_recycle=3600,            # Recycle connections every hour
            pool_size=5,                  # Maximum 5 connections in pool
            max_overflow=10,              # Allow up to 10 overflow connections
            pool_timeout=30               # Timeout after 30 seconds
        )
        print("✅ PostgreSQL database engine created successfully")
    except Exception as e:
        print(f"⚠️ Database engine creation failed: {e}")
        engine = None
else:
    print("⚠️ DATABASE_URL not set - database features disabled")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

# Templates setup
templates = Jinja2Templates(directory="templates")

# Authentication functions
def create_session_token():
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_session(request: Request) -> bool:
    session_token = request.cookies.get("session_token")
    return session_token in active_sessions

def require_auth(request: Request):
    if not verify_session(request):
        raise HTTPException(status_code=401, detail="Authentication required")

def migrate_database():
    """Add missing columns to existing tables"""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            # Check if remaining_value column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='vouchers' AND column_name='remaining_value';
            """))
            if not result.fetchone():
                # Add remaining_value column
                conn.execute(text("ALTER TABLE vouchers ADD COLUMN remaining_value FLOAT;"))
                # Set remaining_value to the same as value for existing vouchers
                conn.execute(text("UPDATE vouchers SET remaining_value = value WHERE remaining_value IS NULL;"))
                # Make remaining_value NOT NULL
                conn.execute(text("ALTER TABLE vouchers ALTER COLUMN remaining_value SET NOT NULL;"))
                conn.commit()
                print("✅ Added remaining_value column to vouchers table")
            else:
                print("✅ remaining_value column already exists")
    except Exception as e:
        print(f"❌ Migration error: {e}")

# Enhanced Database Session Management with Auto-Disconnect
def get_db():
    """Enhanced database session with automatic cleanup and error handling"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"Database connection error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Auto-cleanup for idle connections
async def cleanup_idle_connections():
    """Automatically cleanup idle database connections"""
    while True:
        try:
            # Dispose all connections in pool every 30 minutes
            engine.dispose()
            print("🔄 Database connections cleaned up")
        except Exception as e:
            print(f"Connection cleanup error: {e}")
        await asyncio.sleep(1800)  # 30 minutes

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    credits = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    transactions = relationship("Transaction", back_populates="user")

class Voucher(Base):
    __tablename__ = "vouchers"
    __table_args__ = (UniqueConstraint('code', name='uq_voucher_code'),)

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, nullable=False, index=True)
    value = Column(Float, nullable=False)              # original voucher value
    remaining_value = Column(Float, nullable=False)    # how much left
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float, nullable=False)  # positive=credit; negative=deduction
    type = Column(String, nullable=False)   # e.g. 'redeem', 'claim_deduction', 'claim_fail'
    meta = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="transactions")

# Create tables and run migrations (only if database is available)
if engine:
    try:
        Base.metadata.create_all(bind=engine)
        migrate_database()
        print("✅ Database tables created and migrations applied")
    except Exception as e:
        print(f"⚠️ Database setup failed: {e}")
else:
    print("⚠️ Skipping database operations - engine not available")

app = FastAPI()

# Add CORS middleware to allow Stake domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TM runs on stake.* and talks to Render backend
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to login page"""
    if verify_session(request):
        return RedirectResponse(url="/dashboard")
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    """Display Chinese login page"""
    if verify_session(request):
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login authentication"""
    if username == ADMIN_USERNAME and hash_password(password) == hash_password(ADMIN_PASSWORD):
        session_token = create_session_token()
        active_sessions.add(session_token)
        
        response = RedirectResponse(url="/dashboard", status_code=302)
        response.set_cookie(
            key="session_token", 
            value=session_token, 
            httponly=True, 
            secure=True, 
            samesite="lax"
        )
        return response
    else:
        return RedirectResponse(url="/login?error=用户名或密码错误", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Protected dashboard page"""
    require_auth(request)
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session"""
    session_token = request.cookies.get("session_token")
    if session_token:
        active_sessions.discard(session_token)
    
    response = RedirectResponse(url="/login")
    response.delete_cookie("session_token")
    return response

# Static files mount removed since index.html is in root directory

class RedisPublisher:
    """Simplified Redis-only publisher - WebSocket handling moved to Node.js"""
    def __init__(self):
        self.code_ownership: Dict[str, str] = {}  # code -> username for tracking
    def publish_to_redis(self, message: Dict[str, Any]):
        """Publish message to Redis - Node.js will handle client broadcasting"""
        # Add timestamp for tracking
        message["server_ts"] = int(asyncio.get_event_loop().time() * 1000)

        # Track code ownership
        if message.get("type") == "code":
            code = message.get("code")
            if code:
                self.code_ownership[code] = message.get("username", "system")
            
            # ⬇️ REDIS PUB/SUB: Publish to Redis, Node.js Socket.IO will broadcast to clients
            if redis_manager:
                try:
                    # Publish to 'bonus_codes' channel that Node.js is listening to
                    subscribers = redis_manager.publish_code(message)
                    print(f"📡 Code {code} published to Redis → Node.js will broadcast to {subscribers} subscribers")
                    return True
                except Exception as e:
                    print(f"⚠️ Redis publish failed: {e}")
                    return False
            else:
                print("⚠️ Redis not available - message not published")
                return False
        return True

    def validate_code_ownership(self, code: str, username: str) -> bool:
        """Check if the code belongs to the specified username"""
        return self.code_ownership.get(code) == username

redis_publisher = RedisPublisher()
ring: List[Dict[str, Any]] = []
seen: Set[str] = set()

# Initialize keep-alive service
keep_alive_service = KeepAliveService(RENDER_URL, KEEP_ALIVE_INTERVAL) if KEEP_ALIVE_ENABLED else None

# Telegram Bot Setup
api_id = 2040
api_hash = "b18441a1ff607e10a989891a5462e627"  # official Telethon defaults
bot = None
if TELEGRAM_BOT_TOKEN:
    bot = TelegramClient("voucher-bot", api_id, api_hash)

def generate_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))

async def setup_bot_handlers():
    """Setup bot event handlers after bot is started"""
    if not bot:
        return
        
    @bot.on(events.NewMessage(pattern=r"^/voucher (\d+(\.\d+)?)$"))
    async def handler(event):
        if event.sender_id != TELEGRAM_ADMIN_ID:
            await event.reply("⛔ You are not authorized to use this bot.")
            return

        value = float(event.pattern_match.group(1))
        code = generate_code()
        
        db = SessionLocal()
        try:
            voucher = Voucher(code=code, value=value, remaining_value=value)
            db.add(voucher)
            db.commit()
            db.refresh(voucher)
            await event.reply(f"✅ Voucher Created!\n\n💳 Code: `{voucher.code}`\n💰 Value: {voucher.value}")
        except Exception as e:
            db.rollback()
            await event.reply(f"❌ Error creating voucher: {str(e)}")
        finally:
            db.close()

def normalize_code(s: str) -> str:
    # Remove non-alphanumeric characters but preserve original case
    return re.sub(r"[^A-Za-z0-9]", "", s)

def extract_codes_with_values(text: str) -> List[Dict[str, Any]]:
    """Extract bonus codes and values using multiple patterns, prioritizing 'Code:' format"""
    if not text:
        return []
    print(f"🔍 Input text: {repr(text)}")  # Debug print to see exact text

    all_codes = []

    # First try the CODE_VALUE_PATTERN to extract code and value together
    try:
        pattern = re.compile(CODE_VALUE_PATTERN, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        matches = pattern.findall(text)
        print(f"🔍 Code+Value pattern -> Found: {matches}")  # Debug print

        for match in matches:
            if isinstance(match, tuple) and len(match) >= 1:
                code = match[0].strip()
                value = match[1] if len(match) > 1 and match[1] else None
                if code:
                    all_codes.append({"code": code, "value": value})
                    print(f"🎯 Found code with value: {code} = ${value}" if value else f"🎯 Found code: {code}")
    except Exception as e:
        print(f"⚠️ Code+Value pattern error: {e}")

    # Then try all regular patterns (only for codes without values)
    for i, pattern_str in enumerate(CODE_PATTERNS):
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
            matches = pattern.findall(text)

            print(f"🔍 Pattern {i+1}: {pattern_str} -> Found: {matches}")  # Debug print
            # Handle both string and tuple results
            for match in matches:
                if isinstance(match, tuple):
                    for group in match:
                        if group:
                            code = group.strip()
                            # Check if this code was already found with a value
                            already_exists = any(existing["code"] == code for existing in all_codes)
                            if not already_exists:
                                all_codes.append({"code": code, "value": None})
                else:
                    code = match.strip()
                    # Check if this code was already found with a value
                    already_exists = any(existing["code"] == code for existing in all_codes)
                    if not already_exists:
                        all_codes.append({"code": code, "value": None})
        except Exception as e:
            print(f"⚠️ Pattern error for {pattern_str}: {e}")
            continue

    print(f"🔍 All extracted codes before filtering: {all_codes}")  # Debug print

    # Filter and validate codes
    valid_codes = []
    for item in all_codes:
        code = item["code"]
        value = item["value"]
        # Remove any non-alphanumeric characters
        cleaned_code = re.sub(r"[^A-Za-z0-9]", "", code)

        # Valid codes: 4-25 characters, alphanumeric
        if 4 <= len(cleaned_code) <= 25 and cleaned_code.isalnum():
            valid_codes.append({"code": cleaned_code, "value": value})

    # Remove duplicates while preserving order
    unique_codes = []
    for item in valid_codes:
        code = item["code"]
        if not any(existing["code"] == code for existing in unique_codes):
            unique_codes.append(item)

    print(f"🔍 Final extracted codes: {unique_codes}")
    return unique_codes

def extract_codes(text: str) -> List[str]:
    """Legacy function for backward compatibility - extracts only codes"""
    codes_with_values = extract_codes_with_values(text)
    return [item["code"] for item in codes_with_values]

def ring_add(entry: Dict[str, Any]):
    ring.append(entry)
    if len(ring) > RING_SIZE:
        ring.pop(0)

def ring_latest() -> Optional[Dict[str, Any]]:
    return ring[-1] if ring else None

def ring_get_all() -> List[Dict[str, Any]]:
    """Get all entries from the ring buffer"""
    return list(ring)

tg_client = None
telegram_connected = False # Initialize to False

async def force_disconnect_all():
    """Force disconnect any existing Telegram connections"""
    global tg_client
    try:
        if tg_client:
            print("🔄 Force disconnecting existing Telegram client...")
            try:
                if tg_client.is_connected():
                    await asyncio.wait_for(tg_client.disconnect(), timeout=5.0)
                    print("✅ Previous connection disconnected successfully")
            except Exception as e:
                print(f"⚠️ Error disconnecting previous client: {e}")
            finally:
                tg_client = None
    except Exception as e:
        print(f"⚠️ Force disconnect error: {e}")

async def check_connection_health():
    """Check if Telegram connection is healthy and reconnect if needed"""
    global tg_client, telegram_connected
    try:
        if not tg_client or not tg_client.is_connected():
            print("⚠️ Connection health check failed - connection lost")
            telegram_connected = False
            return False
        
        # Test the connection with a simple API call
        try:
            await asyncio.wait_for(tg_client.get_me(), timeout=5.0)
            return True
        except Exception as e:
            print(f"⚠️ Connection health check failed: {e}")
            telegram_connected = False
            return False
    except Exception as e:
        print(f"⚠️ Error during connection health check: {e}")
        return False

async def ensure_tg(force_reconnect=False):
    global tg_client
    
    # If force reconnect is requested, disconnect first
    if force_reconnect:
        await force_disconnect_all()
    
    # Check if we already have a connected client
    if tg_client and tg_client.is_connected() and not force_reconnect:
        print("✅ Using existing Telegram connection")
        return tg_client
    
    print(f"🔐 Creating Telegram client with session: {TG_SESSION}")
    print(f"🔐 API_ID: {TG_API_ID}")
    print(f"🔐 API_HASH: {'*' * len(TG_API_HASH) if TG_API_HASH else 'Not set'}")
    
    # Retry logic for connection
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Force disconnect any previous connections first
            await force_disconnect_all()
            
            client = TelegramClient(TG_SESSION, TG_API_ID, TG_API_HASH)
            print(f"🔌 Attempting to connect to Telegram (attempt {attempt + 1}/{max_retries})...")
            
            # Add timeout to prevent hanging
            await asyncio.wait_for(client.connect(), timeout=15.0)
            print("✅ Connected to Telegram successfully!")
            
            print("🔍 Checking authorization status...")
            is_authorized = await asyncio.wait_for(client.is_user_authorized(), timeout=10.0)
            
            if not is_authorized:
                print("❌ Session not authorized!")
                phone = os.getenv("TG_PHONE")
                login_code = os.getenv("TG_LOGIN_CODE")
                if not phone or not login_code:
                    raise RuntimeError("Session not authorized. You need to create a session file locally first. See RENDER_DEPLOYMENT_GUIDE.md")
                print(f"📱 Attempting login with phone: {phone}")
                await asyncio.wait_for(client.send_code_request(phone), timeout=15.0)
                try:
                    await asyncio.wait_for(client.sign_in(phone=phone, code=login_code), timeout=15.0)
                    print("✅ Signed in successfully!")
                except SessionPasswordNeededError:
                    print("🔐 2FA required...")
                    pw = os.getenv("TG_2FA_PASSWORD")
                    if not pw:
                        raise RuntimeError("2FA required. Set TG_2FA_PASSWORD environment variable.")
                    await asyncio.wait_for(client.sign_in(password=pw), timeout=15.0)
                    print("✅ 2FA authentication successful!")
            else:
                print("✅ Session already authorized!")
                
            tg_client = client
            print("🎉 Telegram client setup complete!")
            return client
            
        except asyncio.TimeoutError:
            print(f"❌ TELEGRAM CONNECTION TIMEOUT (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("❌ All connection attempts failed")
                raise RuntimeError("Telegram connection timeout after all retries")
                
        except Exception as e:
            error_msg = str(e).lower()
            print(f"❌ TELEGRAM CLIENT ERROR (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"❌ Error type: {type(e).__name__}")
            
            # Handle specific "already connected" errors
            if "already" in error_msg and "connect" in error_msg:
                print("🔄 'Already connected' error detected - forcing reconnection...")
                await force_disconnect_all()
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
            
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise e

async def start_listener():
    global telegram_connected
    max_connection_retries = 5
    
    for connection_attempt in range(max_connection_retries):
        try:
            print(f"🚀 Starting Telegram listener (attempt {connection_attempt + 1}/{max_connection_retries})...")
            
            # Force fresh connection on retry attempts
            force_reconnect = connection_attempt > 0
            
            try:
                client = await asyncio.wait_for(ensure_tg(force_reconnect=force_reconnect), timeout=30.0)
                if not client:
                    print("❌ Failed to ensure Telegram client. Listener not started.")
                    if connection_attempt < max_connection_retries - 1:
                        print(f"🔄 Retrying connection in 3 seconds...")
                        await asyncio.sleep(3)
                        continue
                    return
            except asyncio.TimeoutError:
                print(f"❌ TELEGRAM SETUP TIMEOUT (attempt {connection_attempt + 1}/{max_connection_retries})")
                if connection_attempt < max_connection_retries - 1:
                    print("🔄 Retrying with force reconnect...")
                    await asyncio.sleep(3)
                    continue
                else:
                    print("❌ All connection attempts failed - continuing without Telegram")
                    telegram_connected = False
                    return
            except Exception as e:
                error_msg = str(e).lower()
                print(f"❌ Connection error (attempt {connection_attempt + 1}): {e}")
                
                # Handle "already connected" or similar errors
                if "already" in error_msg or "connect" in error_msg:
                    print("🔄 Connection conflict detected - forcing fresh connection...")
                    await force_disconnect_all()
                    await asyncio.sleep(2)
                    if connection_attempt < max_connection_retries - 1:
                        continue
                
                if connection_attempt < max_connection_retries - 1:
                    print(f"🔄 Retrying in 3 seconds...")
                    await asyncio.sleep(3)
                    continue
                else:
                    telegram_connected = False
                    return
            
            # Successfully connected - proceed with setup
            print(f"🎯 Current CHANNELS environment variable: '{CHANNELS}'")
            print(f"🎯 CHANNELS type: {type(CHANNELS)}")
            # Parse multiple channels from comma-separated string
            channel_list = [ch.strip() for ch in CHANNELS.split(',') if ch.strip()]
            print(f"📋 Parsed {len(channel_list)} channels: {channel_list}")
            # Channel mapping for better tracking
            channel_names = {}
            # First, let's list available dialogs to help find the correct channels
            print("🔍 Listing your available chats/channels:")
            async for dialog in client.iter_dialogs():
                print(f"  📱 {dialog.name} (ID: {dialog.id}, Username: {getattr(dialog.entity, 'username', 'None')})")
            # Validate each channel and build list of valid channels
            valid_channels = []
            for i, channel in enumerate(channel_list):
                try:
                    print(f"🔍 [{i+1}/{len(channel_list)}] Attempting to get entity for: {channel}")
                    entity = await client.get_entity(int(channel) if channel.lstrip('-').isdigit() else channel)
                    channel_name = entity.title if hasattr(entity, 'title') else (entity.first_name or f"Channel_{entity.id}")
                    channel_names[str(entity.id)] = channel_name
                    print(f"✅ [{i+1}] Successfully connected to: {channel_name} (ID: {entity.id})")
                    # Convert to proper format for event listener
                    channel_for_events = int(channel) if channel.lstrip('-').isdigit() else channel
                    valid_channels.append(channel_for_events)
                except Exception as e:
                    print(f"❌ [{i+1}] Could not access channel {channel}: {e}")
                    print(f"   Make sure you're a member of this channel/chat")
                    print("   Skipping this channel...")
            if not valid_channels:
                print("❌ No valid channels found. Please check your CHANNELS environment variable")
                return
            print(f"🎯 Setting up event listener for {len(valid_channels)} channels: {valid_channels}")
            
            @client.on(events.NewMessage(chats=valid_channels))
            async def handler(ev):
                # PRIORITY: Process message immediately without any delays
                channel_name = channel_names.get(str(ev.chat_id), f"Unknown_{ev.chat_id}")
                # ULTRA-FAST: Minimal logging for speed
                print(f"⚡ INSTANT MESSAGE from {channel_name}")
                # Get message text
                text = (ev.message.message or "")
                # Add caption if it exists (for media messages)
                if hasattr(ev.message, 'caption') and ev.message.caption:
                    text += "\n" + ev.message.caption
                # ULTRA-FAST: Extract codes immediately
                codes_with_values = extract_codes_with_values(text)
                if codes_with_values:
                    print(f"⚡ FOUND {len(codes_with_values)} codes")
                if not codes_with_values:
                    print("❌ No valid codes found in message")
                    return
                ts = int(ev.message.date.timestamp() * 1000)
                broadcast_count = 0
                entries_to_broadcast = []
                
                # Process all codes first
                for item in codes_with_values:
                    code = item["code"]
                    value = item["value"]
                    if code in seen:
                        print(f"⚠️ Code {code} already seen, skipping")
                        continue
                    seen.add(code)
                    # Enhanced entry with more metadata
                    entry = {
                        "type": "code",
                        "code": code,
                        "ts": ts,
                        "msg_id": ev.message.id,
                        "channel": str(ev.chat_id),
                        "channel_name": channel_name,
                        "claim_base": CLAIM_URL_BASE,
                        "priority": "instant",
                        "telegram_ts": ts,
                        "broadcast_ts": int(asyncio.get_event_loop().time() * 1000),
                        "source": "telegram",
                        "message_preview": text[:100],
                        "username": "system"  # Default to system since no username yet
                    }

                    # Add value to entry if it exists
                    if value:
                        entry["value"] = value
                    ring_add(entry)
                    entries_to_broadcast.append(entry)
                    broadcast_count += 1
                    print(f"⚡ PREPARED #{broadcast_count}: {code}")
                
                # Broadcast all codes simultaneously
                if entries_to_broadcast:
                    # Create all broadcast tasks at once for true simultaneous sending
                    # Publish all codes to Redis for Node.js to broadcast
                    for entry in entries_to_broadcast:
                        redis_publisher.publish_to_redis(entry)
                    print(f"⚡ BROADCASTING {len(broadcast_tasks)} codes SIMULTANEOUSLY to all clients")
                    # Optional: await all tasks to ensure they complete
                    # await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            
            # Update global status
            telegram_connected = True
            print("🎉 Event listener setup complete! Ready to receive messages...")
            print("🚀 Starting Telegram client and listening for messages...")
            await client.start()
            print("✅ Telegram client started successfully!")
            await client.run_until_disconnected()
            
            # Successfully completed - break out of retry loop
            break
            
        except Exception as e:
            print(f"❌ LISTENER ERROR: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            telegram_connected = False

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return FileResponse('index.html')

@app.get("/api")
@app.head("/api")
async def api_root():
    return JSONResponse({"status": "running", "endpoints": ["/health", "/latest", "/version", "/ws"]})

@app.on_event("startup")
async def startup_event():
    # ⬇️ add at the top of startup_event():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables are ensured (create_all).")
    except Exception as e:
        print(f"❌ DB init error: {e}")
    
    # Start automatic connection cleanup
    print("🔄 Starting automatic database connection cleanup...")
    asyncio.create_task(cleanup_idle_connections())
    
    print("🚀 STAKE ULTRA CLAIMER - RENDER DEPLOYMENT")
    print("=" * 50)
    print(f"📡 Server starting on port {PORT}")
    print(f"🔑 TG_API_ID configured: {'✅' if TG_API_ID else '❌'}")
    print(f"🔑 TG_API_HASH configured: {'✅' if TG_API_HASH else '❌'}")
    print(f"📺 CHANNELS configured: {'✅' if CHANNELS else '❌'}")
    if TG_API_ID and TG_API_HASH and CHANNELS:
        channel_count = len([ch.strip() for ch in CHANNELS.split(',') if ch.strip()])
        print(f"🚀 Starting Telegram listener for {channel_count} channels: {CHANNELS}")
        print("⚡ Ultra-fast code extraction and broadcasting enabled")
        asyncio.create_task(start_listener())
    else:
        print("❌ STARTUP ERROR: Missing Telegram configuration")
        print("   Required environment variables:")
        print("   - TG_API_ID (Telegram API ID)")
        print("   - TG_API_HASH (Telegram API Hash)")
        print("   - CHANNELS (Channel IDs separated by comma)")
        print("   Example: CHANNELS=-1002772030545,-1001234567890")
    # Start keep-alive service
    if keep_alive_service:
        print(f"🔄 Starting keep-alive service (ping every {KEEP_ALIVE_INTERVAL} minutes)")
        asyncio.create_task(keep_alive_service.start_keep_alive())
    else:
        print("ℹ️ Keep-alive service disabled")
    
    # Start Telegram bot
    if bot and TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_ID:
        print(f"🤖 Starting Telegram voucher bot...")
        print(f"🔑 Admin ID: {TELEGRAM_ADMIN_ID}")
        async def run_bot():
            try:
                await bot.start(bot_token=TELEGRAM_BOT_TOKEN)
                await setup_bot_handlers()
                await bot.run_until_disconnected()
            except Exception as e:
                print(f"Bot crashed: {e}")
        asyncio.create_task(run_bot())
    else:
        print("ℹ️ Telegram bot disabled - set TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_ID to enable")

def _get_or_create_user(db: Session, username: str) -> User:
    # Validate and sanitize username input at the database function level
    if isinstance(username, dict):
        print(f"⚠️ CRITICAL: _get_or_create_user received dict instead of string: {username}")
        raise ValueError("Username must be a string, not a dictionary")
    
    # Ensure username is a clean string
    username = str(username).strip()
    if not username:
        raise ValueError("Username cannot be empty")
    
    user = db.query(User).filter(User.username == username).first()
    if not user:
        # Give new users $0.5 welcome bonus for testing/trial
        user = User(username=username, credits=0.5)
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"🎁 New user '{username}' created with $0.5 welcome bonus!")
    return user

@app.get("/balance")
async def get_balance(request: Request, username: str, db: Session = Depends(get_db)):
    # Allow API access with valid session or API key
    if not verify_session(request) and request.headers.get("X-API-Key") != WS_SECRET:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Additional validation for username parameter
    if not username or not isinstance(username, str):
        print(f"⚠️ Invalid username in /balance: {username} (type: {type(username)})")
        raise HTTPException(status_code=400, detail="Invalid username parameter")
    
    try:
        user = _get_or_create_user(db, username)
        return {"username": user.username, "credits": round(user.credits, 8)}
    except ValueError as e:
        print(f"⚠️ Username validation error in /balance: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/redeem")
async def redeem(request: Request, body: dict = Body(...), db: Session = Depends(get_db)):
    # Allow API access with valid session or API key
    if not verify_session(request) and request.headers.get("X-API-Key") != WS_SECRET:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    username = body.get("username")
    voucher_code = body.get("voucher_code")
    
    if not username or not voucher_code:
        raise HTTPException(status_code=400, detail="username and voucher_code required")

    user = _get_or_create_user(db, username)

    voucher = db.query(Voucher).filter(Voucher.code == voucher_code).first()
    if not voucher:
        raise HTTPException(status_code=404, detail="Voucher not found")

    if voucher.remaining_value <= 0:
        raise HTTPException(status_code=400, detail="Voucher already fully redeemed")

    # If amount is specified, use partial redemption; otherwise redeem all remaining
    redeem_amount = body.get("amount")
    if redeem_amount is not None:
        redeem_amount = float(redeem_amount)
        if redeem_amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0")
        if redeem_amount > voucher.remaining_value:
            raise HTTPException(status_code=400, detail=f"Only {voucher.remaining_value} left on voucher")
    else:
        # Backward compatibility: redeem all remaining value
        redeem_amount = voucher.remaining_value

    # Deduct from voucher and credit user
    voucher.remaining_value -= redeem_amount
    user.credits += redeem_amount

    transaction_meta = f"Voucher {voucher.code}" + (" partial" if body.get("amount") is not None else " full")
    db.add(Transaction(user_id=user.id, amount=redeem_amount, type="redeem", meta=transaction_meta))
    db.commit()

    return {
        "ok": True,
        "redeemed": redeem_amount,
        "remaining": round(voucher.remaining_value, 8),
        "credits": round(user.credits, 8)
    }

@app.get("/voucher-info")
async def voucher_info(voucher_code: str, db: Session = Depends(get_db)):
    voucher = db.query(Voucher).filter(Voucher.code == voucher_code).first()
    if not voucher:
        raise HTTPException(status_code=404, detail="Voucher not found")

    return {
        "code": voucher.code,
        "total_value": voucher.value,
        "remaining_value": voucher.remaining_value
    }

@app.post("/claim")
async def claim(request: Request, body: dict = Body(...), db: Session = Depends(get_db)):
    """
    Tampermonkey calls this after an attempted claim.
    body = { username, code, value, success }
    """
    # Allow API access with valid session or API key
    if not verify_session(request) and request.headers.get("X-API-Key") != WS_SECRET:
        raise HTTPException(status_code=401, detail="Authentication required")
    username = body.get("username")
    code = body.get("code")
    value = float(body.get("value") or 0)
    success = bool(body.get("success"))

    # Validate and sanitize username input
    if not username:
        raise HTTPException(status_code=400, detail="username required")
    
    # Ensure username is a string, not a dict or other object
    if isinstance(username, dict):
        print(f"⚠️ WARNING: Received dict instead of username string: {username}")
        raise HTTPException(status_code=400, detail="Invalid username format - expected string")
    
    # Convert username to string and validate
    username = str(username).strip()
    if not username or not code:
        raise HTTPException(status_code=400, detail="username and code required")

    user = _get_or_create_user(db, username)

    if not success:
        # Log the failed attempt (no deduction)
        db.add(Transaction(user_id=user.id, amount=0.0, type="claim_fail", meta=f"Code {code}"))
        db.commit()
        return {"success": False, "credits": round(user.credits, 8)}

    # Success path → deduct 4%
    deduction = round(value * 0.04, 8)
    
    # Check if user has sufficient balance (prevent negative balance)
    if user.credits < deduction:
        # Not enough credits, log failed attempt and return recharge message
        db.add(Transaction(user_id=user.id, amount=0.0, type="insufficient_balance", meta=f"Code {code} - Need ${deduction}, Have ${user.credits}"))
        db.commit()
        return {
            "success": False, 
            "credits": round(user.credits, 8),
            "error": "Insufficient balance",
            "message": "Recharge first to use code claimer",
            "required": deduction,
            "available": round(user.credits, 8)
        }
    
    # Deduct fee and ensure balance never goes below 0
    user.credits = max(0, user.credits - deduction)
    db.add(Transaction(user_id=user.id, amount=-deduction, type="claim_deduction", meta=f"Claim {code} from {value}"))
    db.commit()
    db.refresh(user)
    return {"success": True, "credits": round(user.credits, 8)}

@app.get("/health")
@app.head("/health")
async def health():
    # Enhanced health check with database connection test
    try:
        # Quick database connectivity test
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return PlainTextResponse("OK", 200)
    except Exception as e:
        print(f"Health check failed - database error: {e}")
        return PlainTextResponse("Database Connection Failed", 503)

@app.get("/keepalive")
@app.head("/keepalive")
async def keepalive():
    """UptimeRobot monitoring endpoint"""
    return JSONResponse({
        "status": "alive",
        "timestamp": int(asyncio.get_event_loop().time() * 1000),
        "service": "stake-auto-claimer",
        "redis_connected": redis_manager is not None,
        "codes_processed": len(ring),
        "uptime": "online"
    }, 200)

@app.get("/status")
async def status():
    """Detailed status endpoint for monitoring"""
    telegram_status = False
    telegram_error = None
    session_exists = False
    try:
        import os
        session_exists = os.path.exists(f"{TG_SESSION}.session")
        if tg_client:
            telegram_status = tg_client.is_connected() and await tg_client.is_user_authorized()
    except Exception as e:
        telegram_error = str(e)
    # Database connection status check
    database_status = "unknown"
    database_error = None
    connection_pool_info = {}
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        database_status = "connected"
        # Get connection pool info
        connection_pool_info = {
            "pool_size": engine.pool.size(),
            "checked_out": engine.pool.checkedout(),
            "overflow": engine.pool.overflow(),
            "checked_in": engine.pool.checkedin()
        }
    except Exception as e:
        database_status = "failed" 
        database_error = str(e)
    
    return JSONResponse({
        "telegram_client": telegram_status,
        "telegram_error": telegram_error,
        "session_file_exists": session_exists,
        "session_path": f"{TG_SESSION}.session",
        "redis_connected": redis_manager is not None,
        "codes_in_history": len(ring),
        "keep_alive_enabled": KEEP_ALIVE_ENABLED,
        "keep_alive_running": keep_alive_service.is_running if keep_alive_service else False,
        "server_time": int(asyncio.get_event_loop().time() * 1000),
        "channels_configured": CHANNELS,
        "api_id_set": bool(TG_API_ID),
        "api_hash_set": bool(TG_API_HASH and len(TG_API_HASH) > 5),
        "database_status": database_status,
        "database_error": database_error,
        "connection_pool": connection_pool_info
    }, 200)

@app.get("/latest")
async def latest():
    return JSONResponse(ring_latest() or {}, 200)

@app.get("/api/codes")
async def api_codes():
    """API endpoint for userscript to fetch available codes - returns empty array (only WebSocket delivers new codes)"""
    current_time = int(asyncio.get_event_loop().time() * 1000)

    response = {
        "codes": [],
        "latest_updated": current_time,
        "total_codes": 0,
        "method": "socketio_only",
        "message": "Connect to Socket.IO server for real-time codes"
    }

    print(f"📡 API request for codes: directing to use Socket.IO for real-time codes")
    return JSONResponse(response, 200)

@app.get("/version")
async def version():
    return JSONResponse({"v":"1.0.0"}, 200)

@app.post("/test-code")
async def test_code(request: dict):
    """Test endpoint to simulate receiving a Telegram code"""
    test_code = request.get("code", "TEST123")
    if test_code in seen:
        return JSONResponse({"status": "already_seen", "code": test_code}, 200)
    seen.add(test_code)
    entry = {
        "type": "code",
        "code": test_code,
        "ts": int(asyncio.get_event_loop().time() * 1000),
        "msg_id": 999,
        "channel": "test",
        "claim_base": CLAIM_URL_BASE,
        "username": "system"
    }
    ring_add(entry)
    print(f"🧪 Publishing test code to Redis for Node.js to broadcast: {test_code}")
    redis_publisher.publish_to_redis(entry)
    print(f"✅ Test code broadcasted successfully: {test_code}")
    return JSONResponse({"status": "sent", "code": test_code, "active_connections": len(ws_manager.active)}, 200)

@app.get("/send-test-code/{code}")
async def send_test_code_get(code: str):
    """Quick test endpoint to send a code via GET request"""
    if code in seen:
        return JSONResponse({"status": "already_seen", "code": code}, 200)
    seen.add(code)
    entry = {
        "type": "code",
        "code": code,
        "ts": int(asyncio.get_event_loop().time() * 1000),
        "msg_id": 999,
        "channel": "test",
        "claim_base": CLAIM_URL_BASE,
        "username": "system"
    }
    ring_add(entry)
    print(f"🧪 Publishing test code to Redis for Node.js to broadcast: {code}")
    redis_publisher.publish_to_redis(entry)
    print(f"✅ Test code broadcasted successfully: {code}")
    return JSONResponse({"status": "sent", "code": code, "published_to_redis": True}, 200)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of services"""
    print("🛑 Starting application shutdown...")
    
    # Stop keep-alive service
    if keep_alive_service:
        keep_alive_service.stop_keep_alive()
        print("✅ Keep-alive service stopped")
    
    # Disconnect Telegram client properly
    await force_disconnect_all()
    print("✅ Telegram client disconnected")
    
    # Close database connections
    try:
        print("🔄 Closing database connections...")
        await cleanup_idle_connections()
        print("✅ Database connections cleaned up")
    except Exception as e:
        print(f"⚠️ Error during database cleanup: {e}")
    
    print("✅ Application shutdown complete")

# WebSocket endpoints removed - Node.js Socket.IO handles all real-time connections
# Python now only publishes to Redis, Node.js broadcasts to clients

# @app.websocket("/ws/anonymous") - REMOVED
# @app.websocket("/ws") - REMOVED

# All WebSocket functionality moved to Node.js Socket.IO server
# Python backend now only publishes to Redis using redis_publisher.publish_to_redis()

# Server startup
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting server on http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
