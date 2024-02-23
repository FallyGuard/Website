from django.shortcuts import render, redirect, HttpResponse

import requests
from AuthApp.ai import model

user_signup_endpoint = 'http://falldetect.somee.com/api/Patient/SignUp'
get_user_by_id = "http://falldetect.somee.com/api/Patient/GetPatientsById?id="
get_user_by_email = "http://falldetect.somee.com/api/Patient/GetUserByEmail?Email="

emergency_contact_endpoint = "http://falldetect.somee.com/api/Caregiver/SignUp"

# Create your views here.


def WebCam(request):
    model(request.session['user'], request.session['location'])
    return redirect("/services/")

# ================== Home Page ==================
def HomePage(request):
    # Your Code Here
    print("===================== GET =====================")
    print(request.session.get("user")) 
    return render(request, 'home.html', {"user": request.session.get("user")})

# ================== Auth Pages ==================
def SignUp(request):

    if request.method == 'POST':
        print("===================== POST =====================")

        # Loop through all fields in POST to be non-empty
        for key in request.POST:
            if request.POST[key] == "":
                return render(request, 'signup.html', {"error": "All fields are required", "user": request.session.get("user")})
        
        user = {
            "userName": request.POST.get("fullname"),
            "userEmail": request.POST.get("email"),
            "userDOB": request.POST.get("dob"),
            "userGender": request.POST.get("gender"),
            "userPhoneNumber": request.POST.get("phone"),
            "password": request.POST.get("password"),
            "country": request.POST.get("country"),
            "city": request.POST.get("city"),
        }

        response = requests.post(user_signup_endpoint, json=user, headers={"Content-Type": "application/json"})

        if response.status_code != 200 :
            return render(request, 'signup.html', {"error": response.text, "user": request.session.get("user")})

        print("============ User created successfully ============")
        
        return redirect(f"/add-emergency-contact?userId={response.json()['id']}")

    return render(request, 'signup.html', {"user": request.session.get("user")})

def AddEmergencyContact(request):

    # Get User by ID
    print("User ID: ", request.GET.get("userId"))
    response = requests.get(get_user_by_id + request.GET.get("userId"))
    user = response.json()

    if request.method == 'POST':

        full_name = request.POST.get("first_name") + " " + request.POST.get("last_name")

        emergency_contact = {
            "userID": request.GET.get("userId"),
            "contactName": full_name,
            "contactEmail": request.POST.get("email"),
            "contactPassword": "123456", # Default password for emergency contact
            "contactPhoto": "string", # Default photo for emergency contact
            "gender": request.POST.get('gender'),
            "contactPhoneNumber": request.POST.get("phone"),
            "relationship": request.POST.get("relationship"), # Must be Filled
        }

        response = requests.post(emergency_contact_endpoint, json=emergency_contact)
        print("Response StatusCode: ", response.status_code)
        
        if response.status_code != 200:
            print(response.text)
            return render(request, 'add-emergency-contact.html', {"error": response.text, "user": user})
        
        return render(request, "add-emergency-contact.html", {"success": "Emergency contact added successfully", "user": user})

    print(user)
    return render(request, 'add-emergency-contact.html', {"user": user })

def Login(request):

    if request.method == 'POST':
        print("===================== POST =====================")
        
        user = {
            "userEmail": request.POST.get("email"),
            "password": request.POST.get("password"),
        }

        response = requests.get(get_user_by_email + user["userEmail"], headers={"Content-Type": "application/json"})
        
        if (response.headers.get("Content-Type") == "application/json"):
            print(response.json())
        
        if response.status_code != 200 :
            return render(request, 'login.html', {"error": response.text, "user": request.session.get("user")})

        print("============ User logged in successfully ============")
        # User logged in successfully, register the user in the session

        request.session["user"] = response.json()
        request.session['location'] = {
            "latitute": request.POST.get('lat'),
            "longitute": request.POST.get('lng')
        }

        return redirect("/")

    return render(request, 'login.html', {"user": request.session.get("user")})

def Profile(request):
    return render(request, 'profile.html', {"user": request.session.get("user")})

def Logout(request):
    request.session.clear()
    return redirect("/login")

# ================== Static Pages ==================
def About(request):
    return render(request, 'about.html', {"user": request.session.get("user")})

def Services(request):
    return render(request, 'services.html', {"user": request.session.get("user")})

def Teams(request):
    return render(request, 'teams.html', {"user": request.session.get("user")})
