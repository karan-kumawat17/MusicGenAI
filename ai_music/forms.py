from django import forms
from .models import ImageSubmission

# Define a ModelForm class for the ImageSubmission model


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageSubmission  # Linking the form to the ImageSubmission model
        fields = ['image']  # Specifying which fields should be included in the form
        labels = {
            # 'image': 'Upload Image', 
        }
        help_texts = {
            # 'image': 'Select an image file for music conversion.', 
        }
        error_messages = {
            'image': {
                'invalid': "Image files only!",  # Custom error message for invalid input
                'required': "Image file is required!",  # Custom error message if no file is provided
            }
        }
