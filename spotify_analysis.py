#!/usr/bin/env python
# coding: utf-8

import warnings
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import plotly
import plotly.express as px
import plotly.graph_objs as go
from dateutil.parser import parse
from wordcloud import WordCloud
import zipfile

warnings.simplefilter(action='ignore', category=FutureWarning)

def read_json_file(zip_file, json_file):
    with zip_file.open(json_file) as f:
        df = pd.read_json(f)
        return df

def read_data():
    SPOTIFY_DATA_FILE = "data/spotify_data.zip"
    STREAM_HISTORY_FILE = "data/StreamingHistory0.json"
    LIBRARY_FILE = "data/YourLibrary.json"
    EXENDED_HISTORY_PREFIX = "endsong"

    with zipfile.ZipFile(SPOTIFY_DATA_FILE) as zip_file:
        zip_file.extractall("data")

        # Streaming history
        df_stream = pd.read_json(STREAM_HISTORY_FILE)        
        df_stream['Song'] = df_stream['trackName'] + " (" + df_stream['artistName'] + ")" 

        # Library
        df_library = pd.read_json(LIBRARY_FILE)

        # Combine all extended history in a dataframe
        json_files = [f for f in zip_file.namelist() if f.startswith(EXENDED_HISTORY_PREFIX)]
        dfs = [read_json_file(zip_file, json_file) for json_file in json_files]
        df_extended = pd.concat(dfs)
        
        # Relevant fields: platform, ms_played, master_metadata_track_name, 
        # master_metadata_album_artist_name, spotify_track_uri, reason_start
        df_extended = df_extended[['platform', 'ms_played', 'ts', 'master_metadata_track_name', 'master_metadata_album_artist_name', 'spotify_track_uri', 'reason_start']]

    return df_stream, df_library, df_extended

def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return hours, minutes, seconds

def convert_timedelta_minutes(duration):
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = hours * 60 + (seconds % 3600) // 60
    return minutes

def convert_milliseconds_to_minutes(milliseconds):
    """Converts milliseconds to minutes"""
    return round(milliseconds / (1000 * 60), 2)


def convert_timedelta_to_hours_minutes_seconds(td):
    """Converts timedelta to hours, minutes, and seconds"""
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)

def analyze_data():
    """Analyzes the streaming history data"""
    # Read streaming history, library and extended history
    df_stream, df_library, df_extended = read_data()

    # Time listened per day over time in minutes
    df_time_listened_per_day = df_stream[['endTime', 'msPlayed']].copy()
    df_time_listened_per_day['Day'] = df_time_listened_per_day['endTime'].str[:10]
    df_time_listened_per_day = df_time_listened_per_day.groupby('Day').sum('msPlayed').reset_index()
    df_time_listened_per_day['Time'] = df_time_listened_per_day['msPlayed'].apply(convert_milliseconds_to_minutes)
    df_time_listened_per_day = df_time_listened_per_day[['Day', 'Time']]

    # Time listened per month over time
    df_time_listened_per_day['Month'] = pd.to_datetime(df_time_listened_per_day['Day']).dt.to_period('M')
    df_time_listened_per_month = df_time_listened_per_day.groupby('Month').sum('Time').reset_index()
    df_time_listened_per_month['Time'] = df_time_listened_per_month['Time'].apply(lambda x: round(x / 60))

    # Line chart of minutes listened to music per day
    fig1 = go.Figure(data=go.Scatter(x=df_time_listened_per_day['Day'].astype(dtype=str), 
                                    y=df_time_listened_per_day['Time'],
                                    marker_color='indianred', text="Minutes"))
    fig1.update_layout({"title": 'Minutes listened to music per day',
                    "xaxis": {"title":"Day"},
                    "yaxis": {"title":"Total minutes"}})
    fig1.update_traces(mode="lines", hovertemplate=None)
    fig1.update_layout(hovermode="x")


    # Line chart of minutes listened to music per month
    fig2 = go.Figure(data=go.Scatter(x=df_time_listened_per_month['Month'].astype(dtype=str), 
                                    y=df_time_listened_per_month['Time'],
                                    marker_color='green', text="Hours"))
    fig2.update_layout({"title": 'Hours listened to music per month',
                    "xaxis": {"title":"Month"},
                    "yaxis": {"title":"Total Hours"}})
    fig2.update_traces(mode="markers+lines", hovertemplate=None)
    fig2.update_layout(hovermode="x")


    # Heat map of hours listened to music per month
    fig3 = px.density_heatmap(df_time_listened_per_month, x=df_time_listened_per_day['Day'].astype(dtype=str), 
                                    y=df_time_listened_per_day['Time'])
    fig3.update_layout({"title": 'Hours listened to music per month',
                    "xaxis": {"title":"Month"},
                    "yaxis": {"title":"Total Hours"}})
    

    # Time listened by hour over time
    df_time_listened_by_hour=df_stream.copy()

    for i, row in df_time_listened_by_hour.iterrows():
        time = datetime.timedelta(milliseconds=float(df_time_listened_by_hour.at[i,'msPlayed']))
        minutes = convert_timedelta_minutes(time)
        df_time_listened_by_hour.at[i,'Time']= minutes

    df_hour=df_time_listened_by_hour.copy()
    df_hour['Hour']=df_time_listened_by_hour['endTime'].str[11:13]
    df_hour = df_hour.sort_values(by=['Hour'],ascending=True)


    # Heat map of music listened by hour of day over time
    fig4 = px.density_heatmap(df_time_listened_per_month, x=df_hour['Hour'].astype(dtype=str), 
                                    y=df_hour['Time'])
    fig4.update_layout({"title": 'Music listened by hour of day',
                    "xaxis": {"title":"Hour"},
                    "yaxis": {"title":"Song duration"}})
    fig4.update(layout_yaxis_range = [0,4])
    

    # Time listened by hour of day over time
    df_time_listened_by_hour=df_stream.copy()
    df_time_listened_by_hour['Day']=df_time_listened_by_hour['endTime'].str[11:13]

    df_time_listened_by_hour=df_time_listened_by_hour[['Day', 'msPlayed']]\
        .groupby('Day')\
        .count()\
        .sort_values('msPlayed', ascending=False)\
        .reset_index()

    df_time_listened_by_hour.columns = ['Time of Day', 'Count']


    # Bar chart of songs listened by hour of day
    df_time_listened_by_hour = df_time_listened_by_hour.sort_values(by=['Time of Day'],ascending=True)
    fig5 = px.bar(df_time_listened_by_hour, x=df_time_listened_by_hour['Time of Day'].astype(dtype=str), 
                                    y=df_time_listened_by_hour['Count'])
    fig5.update_layout({"title": 'Number of songs listened by the time of the day in 2022',
                    "xaxis": {"title":"Time of Day"},
                    "yaxis": {"title":"Number of songs listened"}})


    # Minutes listened and song count per artist (Top 100)
    df_artists=df_stream[['artistName', 'msPlayed']]\
        .groupby('artistName')\
        ['msPlayed'].agg(msPlayed='sum', Count='count')\
        .sort_values('msPlayed', ascending=False)\
        .reset_index()
    df_artists.columns = ['Artist', 'msPlayed', 'Songs count']

    for i, row in df_artists.iterrows():
        time = datetime.timedelta(milliseconds=float(df_artists.at[i,'msPlayed']))
        hours, minutes, seconds = convert_timedelta(time)
        df_artists.at[i,'Time']= '{} hours, {} minutes'.format(hours, minutes)

    df_artists = df_artists[['Artist', 'Time', 'Songs count']]
    df_artists = df_artists[df_artists['Artist']!='Unknown Artist']


    # converted df to dict and plot wordcloud
    artist_freq = dict(zip(df_artists['Artist'].tolist(), df_artists['Songs count'].tolist()))
    wc = WordCloud(background_color='white',width=800, height=400, max_words=200).generate_from_frequencies(artist_freq)
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("static/wordcloud.jpg")

    # Draw bar chart of time played per artist (Top 25)
    df_artists = df_stream[['artistName', 'msPlayed']]\
        .groupby('artistName')\
        ['msPlayed'].agg(msPlayed='sum', Count='count')\
        .sort_values('msPlayed', ascending=False)\
        .reset_index().head(25)
    df_artists.columns = ['Artist', 'msPlayed', 'Songs count']

    for i, row in df_artists.iterrows():
        time = datetime.timedelta(milliseconds=float(df_artists.at[i,'msPlayed']))
        hours, minutes, seconds = convert_timedelta(time)
        df_artists.at[i,'Time']= hours
    df_artists = df_artists[~df_artists['Artist'].str.contains("Unknown Artist")]
    
    fig6 = px.bar(data_frame=df_artists[['Artist', 'Time']],x='Artist', y='Time', 
                height=600, width=800,
                color='Time',
                title="Top 25 Artists with Highest Amount of Time Played",
                labels={'Artist': 'Artist', 'Time': 'Time Played (Hours)'})
    fig6.update_layout(barmode='group', xaxis_tickangle=-45)
    

    # Minutes listened and song count per song (Top 100) ordered by time played
    df_songs=df_stream[['Song', 'msPlayed']]\
        .groupby('Song')\
        ['msPlayed'].agg(msPlayed='sum', Count='count')\
        .sort_values('msPlayed', ascending=False)\
        .reset_index()
    df_songs.columns = ['Song', 'msPlayed', 'Times played']

    for i, row in df_songs.iterrows():
        time = datetime.timedelta(milliseconds=float(df_songs.at[i,'msPlayed']))
        hours, minutes, seconds = convert_timedelta(time)
        df_songs.at[i,'Time']= '{} hours, {} minutes'.format(hours, minutes)

    df_songs = df_songs[['Song', 'Time', 'Times played']]
    df_songs = df_songs[df_songs['Song']!='Unknown Track (Unknown Artist)']
    

    # Minutes listened and song count per song (Top 100) ordered by play count
    df_songs.sort_values('Times played', ascending=False).reset_index().head(100)


    # Draw bar chart of Top 200 Songs with Highest Amount of Times Played
    df_artists = df_stream[['Song', 'msPlayed']]\
        .groupby('Song')\
        .count()\
        .sort_values('msPlayed', ascending=False)\
        .reset_index().head(200)
    df_artists.columns = ['Song', 'Times Played']

    df_artists=df_artists.sort_values('Times Played', ascending=False)
    df_artists = df_artists[~df_artists['Song'].str.contains("Unknown Artist")]
    
    fig7 = px.bar(data_frame=df_artists,x='Song', y='Times Played',  
                height=600, width=800,
                color='Times Played',
                title="Top 200 Songs with Highest Amount of Times Played",
                labels={'Song':'Songs'})
    fig7.update_xaxes(visible=True, showticklabels=False)
    

    # Songs listened to vs songs in my library over time  
    # add columns checking if streamed song is in library
    df_in_library = df_stream.copy()
    df_in_library['In Library'] = np.where(df_in_library['trackName'].isin(df_library['track'].tolist()),1,0)
    df_in_library['Not In Library'] = np.where(~df_in_library['trackName'].isin(df_library['track'].tolist()),1,0)
    df_in_library['Total']=df_in_library['In Library']+df_in_library['Not In Library']
    df_in_library['Day']=df_stream['endTime'].str[0:10]

    df_in_library=df_in_library\
        .groupby('Day')\
        .sum(numeric_only=False)\
        .reset_index()
    df_in_library[['Day', 'In Library', 'Not In Library', 'Total']]
    
    plot = go.Figure()
    
    plot.add_trace(go.Scatter(
        name = 'Songs In Library',
        x = df_in_library['Day'],
        y = df_in_library['In Library'], 
        stackgroup='one',
        line=dict(color='#32CD32'),
    )
    )
    
    plot.add_trace(go.Scatter(
        name = 'Songs Not In Library',
        x = df_in_library['Day'],
        y = df_in_library['Not In Library'],
        stackgroup='one',mode='none' 
    )
    )
    plot.update_layout({"title": 'Comparison between songs in library and songs not in library'})

    # Nr of songs and unique songs per artist in history
    df_unique_songs_artist_history=df_stream[['artistName', 'trackName']]\
        .groupby('artistName')\
        .agg(['count', 'nunique'])\
        .reset_index()
    df_unique_songs_artist_history.columns = ['Artist', 'Songs count', 'Unique Songs count']
    df_unique_songs_artist_history.sort_values('Songs count', ascending=False).reset_index()[['Artist', 'Songs count', 'Unique Songs count']]


    # Nr of songs and unique songs per artist in library
    df_unique_songs_artist_library=df_library[['artist', 'track']]\
        .groupby('artist')\
        .nunique()\
        .reset_index()
    df_unique_songs_artist_library.columns = ['Artist', 'Unique Songs count']
    df_unique_songs_artist_library.sort_values(['Unique Songs count'], ascending=False).reset_index()[['Artist', 'Unique Songs count']]


    # Nr of songs and unique songs per artist in extended history
    df_unique_songs_artist_extended=df_extended[['master_metadata_album_artist_name', 'master_metadata_track_name']]\
        .groupby('master_metadata_album_artist_name')\
        .agg(['count', 'nunique'])\
        .reset_index()
    df_unique_songs_artist_extended.columns = ['Artist', 'Songs Count', 'Unique Songs Count']
    df_unique_songs_artist_extended.sort_values('Songs Count', ascending=False).reset_index()[['Artist', 'Songs Count', 'Unique Songs Count']]


    # Listening over time
    # Time listened per day over time
    df_time_extended=df_extended.copy()
    df_time_extended['Day']=df_extended['ts'].str[0:10]

    df_time_extended=df_time_extended[['Day', 'ms_played']]\
        .groupby('Day')\
        .sum()\
        .sort_values('ms_played', ascending=False)\
        .reset_index()

    for i, row in df_time_extended.iterrows():
        time = datetime.timedelta(milliseconds=float(df_time_extended.at[i,'ms_played']))
        minutes = convert_timedelta_minutes(time)
        df_time_extended.at[i,'Time']= minutes

    df_time_extended = df_time_extended[['Day', 'Time']]\
        .groupby('Day')\
        .sum()\
        .sort_values('Day', ascending=True)\
        .reset_index()

    # Time listened per month over time
    df_time_extended['date'] = [parse(date).date() for date in df_time_extended['Day']]
    df_time_extended['Month'] = pd.to_datetime(df_time_extended['date']).dt.to_period('M')

    df_month_extended = df_time_extended[['Month', 'Time']]\
        .groupby('Month')\
        .sum()\
        .sort_values('Month', ascending=True)\
        .reset_index()
    df_month_extended['Time'] = round(df_month_extended['Time'].apply(lambda x: x/60))

    # Draw line chart by day
    fig8 = go.Figure(data=go.Scatter(x=df_time_extended['Day'].astype(dtype=str), 
                                    y=df_time_extended['Time'],
                                    marker_color='indianred', text="Minutes"))
    fig8.update_layout({"title": 'Minutes listened to music per day',
                    "xaxis": {"title":"Day"},
                    "yaxis": {"title":"Total minutes"}})
    fig8.update_traces(mode="lines", hovertemplate=None)
    fig8.update_layout(hovermode="x")
    

    # Draw line chart by month
    fig9 = go.Figure(data=go.Scatter(x=df_month_extended['Month'].astype(dtype=str), 
                                    y=df_month_extended['Time'],
                                    marker_color='green', text="Hours"))
    fig9.update_layout({"title": 'Hours listened to music per month',
                    "xaxis": {"title":"Month"},
                    "yaxis": {"title":"Total Hours"}})
    fig9.update_traces(mode="markers+lines", hovertemplate=None)
    fig9.update_layout(hovermode="x")


    # Group extended history by platform (device)
    df_platform = df_extended.copy()
    df_platform['Day'] = df_extended['ts'].str[0:10]

    df_platform = df_platform[['platform', 'Day']]
    df_platform.loc[df_platform['platform'].str.contains("iOS"), 'platform']='iOS'
    df_platform.loc[df_platform['platform'].str.contains("Android"), 'platform']='Android'
    df_platform.loc[df_platform['platform'].str.contains("OS X"), 'platform']='OS X'
    df_platform.loc[df_platform['platform'].str.contains("Google_Home"), 'platform']='Google Home'
    df_platform.loc[df_platform['platform'].str.contains("Amazon"), 'platform']='Fire TV'
    df_platform.loc[df_platform['platform'].str.contains("web_player"), 'platform']='Web'
    df_platform_count = df_platform['platform'].value_counts().reset_index()
    df_platform_count.rename(columns={'index': 'Platform', 'platform': 'Track Count'}, inplace=True)
    df_platform_count.sort_values(by='Track Count', ascending=False, inplace=True)

    # Draw heat map listened on device by day
    df_platform = df_platform[~df_platform['platform'].str.contains("Web")]
    fig10 = px.density_heatmap(df_time_listened_per_month, x=df_platform['Day'].astype(dtype=str), 
                                    y=df_platform['platform'])
    fig10.update_layout({"title": 'Music listened on a platform by day',
                    "xaxis": {"title":"Day"},
                    "yaxis": {"title":"Platform"}})
    

    # Draw heat map listened on device by weekday
    df_platform['Weekday'] = [parse(date).date().strftime('%A') for date in df_platform['Day']]
    df_platform['weekday'] = [parse(date).date().weekday() for date in df_platform['Day']]
    df_platform=df_platform.sort_values(by='weekday', ascending=True)

    fig11 = px.density_heatmap(df_time_listened_per_month, x=df_platform['Weekday'].astype(dtype=str), 
                                    y=df_platform['platform'])
    fig11.update_layout({"title": 'Music listened on a platform by weekday',
                    "xaxis": {"title":"Weekday"},
                    "yaxis": {"title":"Platform"}})


    # Draw heat map listened on device by time of day
    df_platform=df_extended.copy()
    df_platform['Day']=df_extended['ts'].str[0:10]
    df_platform['Time of Day']=df_extended['ts'].str[11:13]
    df_platform['Weekday'] = [parse(date).date().strftime('%A') for date in df_platform['Day']]
    df_platform['weekday'] = [parse(date).date().weekday() for date in df_platform['Day']]
    df_platform=df_platform.sort_values(by='weekday', ascending=True)

    df_platform=df_platform.sort_values(by=["Time of Day", "platform"], ascending= True)
    df_platform.loc[df_platform['platform'].str.contains("iOS"), 'platform']='iOS'
    df_platform.loc[df_platform['platform'].str.contains("Android"), 'platform']='Android'
    df_platform.loc[df_platform['platform'].str.contains("OS X"), 'platform']='OS X'
    df_platform.loc[df_platform['platform'].str.contains("Google_Home"), 'platform']='Google Home'
    df_platform.loc[df_platform['platform'].str.contains("Amazon"), 'platform']='Fire TV'
    df_platform.loc[df_platform['platform'].str.contains("web_player"), 'platform']='Web'    
    df_platform = df_platform[~df_platform['platform'].str.contains("Web")]
    fig12 = px.density_heatmap(df_time_listened_per_month, x=df_platform['Time of Day'].astype(dtype=str), 
                                    y=df_platform['platform'])
    fig12.update_layout({"title": 'Music listened on a platform by time of day',
                    "xaxis": {"title":"Time of Day"},
                    "yaxis": {"title":"Platform"}})
    

    # Draw heat map listened on a day by weekday
    df_platform=df_platform.sort_values(by=["Time of Day", "weekday"], ascending= True)
    fig13 = px.density_heatmap(df_time_listened_per_month, x=df_platform['Time of Day'].astype(dtype=str), 
                                    y=df_platform['Weekday'])
    fig13.update_layout({"title": 'Music listened on a day by weekday',
                    "xaxis": {"title":"Time of Day"},
                    "yaxis": {"title":"Weekday"}})
    

    df_platform['Time'] = df_platform['Time of Day'].astype(int)
    df_platform['Part'] = pd.cut(df_platform['Time'],
                                    bins=[-1, 5, 12, 17, 21, 23],
                                    labels=['Night', 'Morning', 'Afternoon',
                                            'Evening', 'Night'],
                                    ordered=False)
    day_of_week = df_platform.groupby(['Part', 'Weekday'])\
                            .agg({'ms_played': np.sum})

    day_of_the_week = {'morning': day_of_week.loc['Morning'],
                'afternoon': day_of_week.loc['Afternoon'],
                'evening': day_of_week.loc['Evening'],
                'night': day_of_week.loc['Night']}

    plt.figure(figsize=(20, 6), dpi=60)
    days_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Plot morning times
    plt.bar(day_of_the_week['morning'].index, day_of_the_week['morning']['ms_played'], label='Morning (6 to 12)')
    # Plot afternoon times
    plt.bar(day_of_the_week['afternoon'].index,
            day_of_the_week['afternoon']['ms_played'],
            bottom=day_of_the_week['morning']['ms_played'],
            label='Afternoon (13 to 17)')
    # Plot evening times
    plt.bar(day_of_the_week['evening'].index,
            day_of_the_week['evening']['ms_played'],
            bottom=day_of_the_week['morning']['ms_played']+day_of_the_week['afternoon']['ms_played'],
            label='Evening (18 to 21)')
    # Plot night times
    plt.bar(day_of_the_week['night'].index, day_of_the_week['night']['ms_played'],
            bottom=day_of_the_week['morning']['ms_played'] +
            day_of_the_week['afternoon']['ms_played'] +
            day_of_the_week['evening']['ms_played'],
            label='Night (22 to 5)')
    plt.xticks(range(7), days_labels)
    plt.suptitle('Hours played by day of the week', fontsize=30)
    plt.legend()
    plt.savefig("static/plot.jpg")

    # Return graphs
    graphJSON = {}
    figures = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, plot]

    for i, fig in enumerate(figures):
        graphJSON[i+1] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON