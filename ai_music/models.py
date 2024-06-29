from django.db import models

# Create your models here.


class ImageSubmission(models.Model):
    image = models.ImageField(upload_to='submissions/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image uploaded on {self.created_at}"