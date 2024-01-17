from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView
import pickle
import pandas as pd



def homePageView(request):
    # return request object and specify page.
    return render(request, 'home.html')


def homePost(request):
    # Use request object to extract choice.

    ticket = -999
    gender = -999
    age = -999
    fare = -999

    try:
        # Extract value from request object by control name.
        currentTicket = request.POST['ticket_class']
        currentGender = request.POST['gender']
        currentAge = request.POST['passenger_age']
        currentFare = request.POST['passenger_fare']

        # Crude debugging effort.
        ticket = int(currentTicket)
        gender = int(currentGender)
        age = int(currentAge)
        fare = float(currentFare)
        print("*** Ticket Class: " + str(ticket))
        print("*** Passenger Gender: " + ("Female" if gender == 1 else "Male"))
        print("*** Passenger Age: " + str(age))
        print("*** Passenger Fare: " + str(fare))
    # Enters 'except' block if integer cannot be created.
    except:
        return render(request, 'home.html', {
            'errorMessage': '*** The data submitted is invalid. Please try again.'})
    else:
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('results', kwargs={'ticket': ticket, 'gender': gender, 'age': age, 'fare': fare}, ))


def results(request, ticket, gender, age, fare):
    print("*** Inside reults()")
    # load saved model
    with open('/home/kolssonbcit/helloworld/static/model_pkl', 'rb') as f:
        loadedModel = pickle.load(f)
    with open('/home/kolssonbcit/helloworld/static/sc_x.pkl', 'rb') as f2:
        scaler = pickle.load(f2)

    # Create a single prediction.
    singleSampleDf = pd.DataFrame(columns=['Pclass', 'Sex', 'Fare'])

    currentFare = float(fare)
    print("*** Ticket Class: " + str(ticket))
    print("*** Passenger Gender: " + ("Female" if gender == 1 else "Male"))
    print("*** Passenger Age: " + str(age))
    print("*** Passenger Fare: " + str(currentFare))
    singleSampleDf = singleSampleDf.append({'Pclass':ticket, 'Sex':gender, 'Age':age, 'Fare':currentFare},
                                           ignore_index=True)

    singleSampleDf_scaled = scaler.transform(singleSampleDf)

    singlePrediction = loadedModel.predict(singleSampleDf_scaled)

    print("Single prediction: " + str(singlePrediction))

    return render(request, 'results.html', {'ticket': ticket, 'gender': gender, 'age': age, 'fare': fare,
                                            'prediction': singlePrediction})

