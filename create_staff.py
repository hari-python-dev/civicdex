
from getpass import getpass
from database import create_staff_account

print("===== CivicDex Department Officer Setup =====")

name = input("Enter Officer Name: ").strip()
email = input("Enter Officer Email: ").strip()

departments = [
    "Roads Department",
    "Water Supply Department",
    "Sanitation Department",
    "Drainage Department",
    "Electricity Department",
    "Revenue Department",
    "Public Health Department",
    "Transport Department",
    "Welfare Department"
]

print("\nAvailable Departments:")

for index, department in enumerate(departments, start=1):
    print(f"{index}. {department}")

choice = input("\nSelect Department Number: ").strip()

if not choice.isdigit() or not (1 <= int(choice) <= len(departments)):
    print("Invalid department selection.")

else:
    department = departments[int(choice) - 1]

    password = getpass("Create Officer Password: ")
    confirm_password = getpass("Confirm Password: ")

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
            role="department",
            department=department
        )

        if created:
            print("\nOfficer account created successfully!")
            print("Officer:", name)
            print("Department:", department)
        else:
            print("This email is already registered.")
