from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)
app.config['DEBUG'] = True

# importar el modelo que está en formato pickle
# IMPORTANTE: al subir a PyhtonEveryWhere modificar por la ruta donde esta el modelo en el directorio de ahí
with open('/home/jaimeih/tc_apis/04_APIs/text_model.pkl', 'rb') as f:  # 'rb' es para que lo lea en binario que sino no ejecuta
    model = pickle.load(f)

@app.route('/')
def home():
    return """
<html>
<head>
    <title>API de Análisis de Sentimientos</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        p {
            margin-bottom: 20px;
        }
        code {
            background-color: #f4f4f4;
            padding: 5px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>Bienvenido a la API de Análisis de Sentimientos</h1>
    <p>Esta API utiliza el modelo VADER (Valence Aware Dictionary and sEntiment Reasoner) para analizar el sentimiento de un texto en inglés. Pertenece a la libreria NLTK (Natural Language Tool Kit)</p>
    <p>El modelo VADER es particularmente adecuado para manejar textos provenientes de redes sociales, donde el lenguaje suele ser informal, incluyendo emoticones, acrónimos y otros tipos de jerga.</p>
    <p>Cómo Funciona VADER:</p>
    <ul>
        <li>Léxico Basado: VADER tiene un léxico (lista de palabras) al que se han asignado puntuaciones de valencia. Estas puntuaciones describen qué tan positiva, negativa o neutral es una palabra. Además, VADER está equipado para entender texto con énfasis (por ejemplo, mayúsculas), negaciones y hasta puntuaciones.</li>
        <li>Análisis de Sentimientos: Al procesar un texto, VADER evalúa cada palabra y frase según su léxico y las reglas gramaticales específicas que entiende (por ejemplo, manejo de negaciones como "no es bueno"). Con base en estos datos, calcula puntuaciones para cada texto.</li>
    </ul>
    <p>El resultado del análisis se devuelve en un diccionario JSON que contiene los siguientes campos:</p>
    <ul>
        <li><strong>text</strong>: El texto proporcionado para análisis.</li>
        <li><strong>sentiment</strong>: Un diccionario con las puntuaciones de sentimiento.</li>
    </ul>
    <p>El diccionario de sentimiento contiene los siguientes campos:</p>
    <ul>
        <li><strong>neg</strong>: Puntuación de negatividad.</li>
        <li><strong>neu</strong>: Puntuación de neutralidad.</li>
        <li><strong>pos</strong>: Puntuación de positividad.</li>
        <li><strong>compound</strong>: Una puntuación compuesta que calcula la suma normalizada de todas las puntuaciones de léxico que han sido normalizadas entre -1 (muy negativo) y +1 (muy positivo).</li>
    </ul>
    <div style="background-color: #FFDCA7; padding: 10px; border-radius: 5px;">
        <p style="font-weight: bold;">Atención:</p>
        <p>La diferencia principal entre la puntuación compuesta y las puntuaciones de positivo, neutro y negativo es que las últimas se calculan a partir de las palabras de forma individual, mientras que la puntuación compuesta tiene en cuenta el contexto global del texto y puede asignar diferentes pesos a las palabras en función de su posición y relación con otras palabras circundantes.</p>
        <p>Debido a esta consideración del contexto y las reglas heurísticas aplicadas, es posible que la puntuación compuesta refleje una polaridad más extrema que las puntuaciones individuales, ya que tiene en cuenta la combinación de palabras en toda la frase. Esto puede explicar por qué la puntuación compuesta puede ser muy negativa mientras que las puntuaciones de positivo, neutro y negativo pueden ser más equilibradas entre sí.</p>
    </div>
    </ul>
    <p>Ejemplo de uso del endpoint:</p>
    <code>http://jaimeih.pythonanywhere.com/api/v1/predict?text=I%20love%20Python</code>
    <p>Este ejemplo analiza el texto "I love Python". Copia y pega esta URL en tu navegador para probarlo.</p>
    <p>Recuerda que después de cada palabra en el texto de la URL es necesario añadir un espacio para que el modelo pueda diferenciarlas (puedes hacerlo añadiendo "%20" despu´és de cada palabra).</p>
</body>
</html>
"""

@app.route('/api/v1/predict')
def predict():
    texto = request.args.get('text', None) # llamada get para pillar el texto del user
    
    if texto is None: # Si no se añade texto , se lanza una excepción con un 400 indicando que es necesario añadirlo
        return jsonify({'error': 'Por favor proporciona un texto para analizar usando el parámetro "text". Recuerda a añadir el texto en inglés'}), 400
    else:
        resultado = model.polarity_scores(texto) # la función polarity_scores toma el texto en formato string y lo analiza. luego devuelve el resultado en un diccionario
        return jsonify({'text': texto, 'sentiment': resultado}) # se convierte el resultado en formato json y se devuelve 

@app.route('/api/v1/retrain', methods=['GET'])
def retrain(): # por ahora dejar en caso de necesitar en el futuro
    return "Este modelo no requiere reentrenamiento ya que usa un modelo preentrenado."

if __name__ == '__main__':
    app.run()

