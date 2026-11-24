from getpass import getpass
from database import create_staff_account

print("===== CivicDex Admin Account Setup =====")

name = input("Enter Admin Name: ").strip()
email = input("Enter Admin Email: ").strip()
password = getpass("Create Admin Password: ")
confirm_password = getpass("Confirm Admin Password: ")

if not name or not email or not password:
    print("All fields are required.")

elif "@" not in email:
    print("Enter a valid email.")

elif len(password) < 8:
    print("Password must contain at least 8 characters.")

elif password != confirm_password:
    print("Passwords do not match.")

else:
    created = create_staff_account(
        name=name,
        email=email,
        password=password,
        role="admin",
        department=None
    )

    if created:
        print("Admin account created successfully!")
    else:
        print("This email is already registered.")