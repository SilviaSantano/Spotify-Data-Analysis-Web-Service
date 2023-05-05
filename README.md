# Spotify Data Analysis Web Service
I created this tool to analyse and visualize your personal Spotify streaming data with insightful graphs in Python: [Spotify Data Analysis and Visualization
](https://github.com/SilviaSantano/Spotify-Data-Analysis).

In order to make use of the tool and be able to execute the code more comfortably, I created this lightweight web service based in the Flask framework that allows anyone to simply upload a copy of their data and run all analysis and visualizations by just clicking a button.

Since some visualizations need to make use of the API and the streaming history can be very large, not all of the graphs that can be generated using the [Jupyter notebook](https://github.com/SilviaSantano/Spotify-Data-Analysis) are possible here, because they would take a long time to load, but only those that do not need the API, such as line graphs showing the amount of time listened to music per day and per month, bar graphs of the songs with the highest amount of times played, the number of songs listened in average by the time of the day, heatmaps of the music listened by device and day of the week... and many more.

## Usage
From the downloaded folder:
- Install all package requirements with ```pip install -r requirements.txt```.
- Run the application in Flask with ```FLASK_APP=app.py flask run```
- Open the app in your browser by navigating to ```http://127.0.0.1:5000```

You should see a website like this, where you can upload the zip containing all data and, after pressing the button, the page will fill up with all visualizations.

<img width="1013" alt="Screenshot 2023-05-18 at 17 01 16" src="https://github.com/SilviaSantano/Spotify-Data-Analysis-Web-Service/assets/12804135/585315fa-6d68-483e-8820-fd8b6bbb4384">

## Data privacy

This web service is not deployed anywhere, it can be run in your own machine, which means the sensible data that you use will not leave your computer at any time as the whole analysis takes place locally.