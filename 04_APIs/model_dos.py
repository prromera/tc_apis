import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle

'''

El modelo VADER (Valence Aware Dictionary and sEntiment Reasoner) es una herramienta de análisis de sentimientos que es particularmente buena 
para manejar textos que provienen de redes sociales, donde el lenguaje suele ser informal, incluyendo emoticones, acrónimos y otros tipos de jerga.

Cómo Funciona VADER
Léxico Basado: VADER tiene un léxico (lista de palabras) al que se han asignado puntuaciones de valencia. Estas puntuaciones describen qué tan positiva, 
negativa o neutral es una palabra. Además, VADER está equipado para entender texto con énfasis (por ejemplo, mayúsculas), negaciones y hasta puntuaciones.

Análisis de Sentimientos: Al procesar un texto, VADER evalúa cada palabra y frase según su léxico y las reglas gramaticales específicas que entiende 
(por ejemplo, manejo de negaciones como "no es bueno"). Con base en estos datos, calcula puntuaciones para cada texto.

Interpretación de los Resultados
VADER proporciona cuatro métricas en su análisis de sentimientos:

neg: Puntuación de negatividad, que mide la proporción de palabras en el texto que son negativas.
neu: Puntuación de neutralidad, que mide la proporción de palabras que son neutrales.
pos: Puntuación de positividad, que mide la proporción de palabras en el texto que son positivas.
compound: Una puntuación compuesta que calcula la suma normalizada de todas las puntuaciones de léxico que han sido normalizadas entre -1 (muy negativo) y +1 (muy positivo).

IMPORTANTE:
La diferencia principal entre la puntuación compuesta y las puntuaciones de positivo, neutro y negativo es que las últimas se calculan a partir de las palabras de forma individual, 
mientras que la puntuación compuesta tiene en cuenta el contexto global del texto y puede asignar diferentes pesos a las palabras en función de su posición y relación con otras palabras circundantes.

Debido a esta consideración del contexto y las reglas heurísticas aplicadas, es posible que la puntuación compuesta refleje una polaridad más extrema que las puntuaciones individuales, 
ya que tiene en cuenta la combinación de palabras en toda la frase. 
Esto puede explicar por qué la puntuación compuesta puede ser muy negativa mientras que las puntuaciones de positivo, neutro y negativo pueden ser más equilibradas entre sí.
'''

# IMPORTANTE: imprescindible descargar los recursos de VADER --> tienen las palabras y sus correspondientes puntuaciones
# El modelo en formato pickle no contiene esta info, va a parte
# En PythonEveryWhere se alojan en el directorio 'nltk_data'
nltk.download('vader_lexicon')

# modelo : analizador de sentimientos de VADER
sia = SentimentIntensityAnalyzer()

# textos de prueba para analizar en local
textos = [
    "I love this!",          
    "I feel sad",            
    "Are you stupid?",       
    "This is great!",        
    "I'm feeling neutral.",
    "¿Qué pasa si escribo en español?",
    "I will never be your friend again"
]

# predict a los textos
resultados = []
for texto in textos:
    resultado = sia.polarity_scores(texto)
    resultados.append(resultado)

# resultados de los textos de prueba
for i, resultado in enumerate(resultados):
    print(f"Resultado de la predicción para texto {i+1}:")
    print(resultado)
    print()

# guardar modelo con pickle --> abre un archivo en formato binario con el alias 'f' y la función dump guarda el modelo en el archivo
with open('text_model.pkl', 'wb') as f:
    pickle.dump(sia, f)

