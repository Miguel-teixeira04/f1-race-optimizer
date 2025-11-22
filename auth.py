"""
Simple authentication system for the race optimizer app
"""
import hashlib
import json
import os

# File to store users
USERS_FILE = "users.json"

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def create_user(username, password, email=""):
    """Create a new user"""
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        "password_hash": hash_password(password),
        "email": email,
        "created_at": str(os.path.getmtime(__file__))
    }
    
    save_users(users)
    return True, "User created successfully"

def verify_user(username, password):
    """Verify user credentials"""
    users = load_users()
    
    if username not in users:
        return False
    
    return users[username]["password_hash"] == hash_password(password)

def initialize_default_users():
    """Create default users if none exist"""
    users = load_users()
    
    if not users:
        # Create default admin user
        default_users = {
            "admin": {
                "password_hash": hash_password("admin123"),
                "email": "admin@racing.com",
                "created_at": "2025-11-12"
            },
            "demo": {
                "password_hash": hash_password("demo123"),
                "email": "demo@racing.com",
                "created_at": "2025-11-12"
            }
        }
        save_users(default_users)
        return True
    return False

if __name__ == "__main__":
    # Initialize default users
    if initialize_default_users():
        print("")
       
    else:
        print("Users already exist")

