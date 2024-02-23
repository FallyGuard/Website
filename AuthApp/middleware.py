# middleware.py
from django.shortcuts import redirect
from django.urls import reverse

class AuthMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):

        try:
            # Check if the request is for a specific URL
            if request.path == '/' or request.path == '/webcam/':
                # Check if the user is not authenticated
                if "user" not in request.session:
                    # Redirect to the login page
                    return redirect(reverse('login'))

            # Continue with the regular flow for other URLs
            response = self.get_response(request)
            return response
        except Exception as e:
            print(f"Middleware error: {e}")
            # Handle the exception or let it propagate
