from flask import Flask, request, render_template
import spotify_analysis

app = Flask(__name__)

@app.route('/')
def upload():
    return render_template('upload.html')

@app.route('/analysis', methods=['POST'])
def analysis():
    # Get the uploaded file
    history = request.files['history']

    # Save the file to disk
    history.save('data/spotify_data.zip')

    graphJSON = spotify_analysis.analyze_data()

    # Render the results
    graphJSON_items = {f'graphJSON{i}': graph for i, graph in graphJSON.items()}
    
    return render_template('upload.html', **graphJSON_items)