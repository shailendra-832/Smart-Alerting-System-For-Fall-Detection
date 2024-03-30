from django.shortcuts import render
from django.http import HttpResponse

from main1 import show_output

from fall3 import detect_fall
from notifications import make_phone_call,whatsapp_message
# Create your views here.
def home(request):
    return render(request,"main.html")

result = {}

def page1(request):
    return render(request,"page1.html")

def page2(request):
    global result
    link = request.POST.get('link')
    result = {
        'data' : link,
    }
    request.session['data'] = link

    return render(request,"page2.html")
    

def page3(request):
    global result
    name = request.POST.get('fullname')
    number = request.POST.get('phonenumber')
    link = request.session['data']
    result = {
        'data' :link,
        'data1' :name,
        'data2' :number,
    }
    print(result)
    
    return render(request,"page3.html",result)

def project(request):
    return render(request,"project.html")

def results(request):
    link = request.session['data']
    print(link)
    print("inside results")
    detect_fall()
    make_phone_call()
    whatsapp_message()
    return render(request,"main.html")

def modelsopt(request):
    return render(request,"modelsopt.html")
