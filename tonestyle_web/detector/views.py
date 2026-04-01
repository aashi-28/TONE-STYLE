from django.shortcuts import render
from .predict import analyze_image
from django.core.files.storage import FileSystemStorage

def index(request):
    result = None

    if request.method == "POST" and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        filepath = fs.path(filename)

        tone, undertone, colors = analyze_image(filepath)

        result = {
            "tone": tone,
            "undertone": undertone,
            "colors": ", ".join(colors)
        }

    return render(request, "index.html", {"result": result})