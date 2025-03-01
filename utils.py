from imports import *

# Este código funcionará en CPU sin problemas. Si en el futuro activas CUDA, también lo usará automáticamente.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Usando: {device}")


#### PROMPT BÁSICOS
prompt1 = "My name"
prompt2 = "Today is"
prompt3 = "Me llamo"

text = """The Second World War (also written World War II)1 was a global military \
conflict that took place between 1939 and 1945. It involved most of the world's \
nations - including all the major powers, as well as virtually all European nations \
- grouped into two opposing military alliances: the Allies on the one hand, and the \
Axis Powers on the other. It was the greatest war in history, with more than 100 \
million military personnel mobilized and a state of total war in which the major \
contenders devoted all their economic, military and scientific capabilities to the \
service of the war effort, blurring the distinction between civilian and military \
resources."""

prompt4 = f"Summarize: {text}"

prompt5 = "What do you think of Mars?"

review1 = """Love these plugs, have a few now. We use them to plug in lights and \
set timers to turn them on and off via a phone app. Easy to use and linked to \
the internet and apps. Good value for money."""

review2 = """Tried and tried but could never get them to work right. Too bad \
I'm past my return date or they would have gone back."""

review3 = """A well-sized, reliable smart plug. The app is easy to use and set \
up, and works well. I used them to make several lamps. Everything works fine - \
no problems."""

review4 = """Great little product. Super easy to set up. Didn't even need to use \
the Alexa app to do so. Did it with my echo. Now I use it almost daily to turn on \
a light that was a pain to get to."""

review5 = """If I could give this zero stars I would. Plug wouldn’t connect. I \
had to keep connecting it and finally just gave up and returned it. Customer service \
was a complete waste of time."""

prompt6 = f"""
Review: {review1}
Sentiment: Positive

Review: {review2}
Sentiment: Negative

Review: {review3}
Sentiment: Positive

Review: {review5}
Sentiment:"""


text1 = """Astrónomos detectaron una misteriosa ráfaga de ondas de radio que tardó \
8.000 millones de años en llegar a la Tierra. La ráfaga rápida de radio es una de \
las más distantes y energéticas jamás observadas. Las ráfagas rápidas de radio \
(FRB, por sus siglas en inglés) son intensos estallidos de ondas de radio de \
milisegundos de duración cuyo origen se desconoce. La primera FRB se descubrió \
en 2007 y, desde entonces, se han detectado cientos de estos rápidos destellos \
cósmicos procedentes de puntos distantes de todo el universo."""

text2 = """"La Revolución Industrial, que tuvo lugar principalmente en el siglo XIX, \
fue un período de grandes cambios tecnológicos, culturales y socioeconómicos que \
transformó a las sociedades agrarias en sociedades industriales. Durante este tiempo, \
hubo un cambio masivo de mano de obra de las granjas a las fábricas. Esto se debió a \
la invención de nuevas máquinas que podían realizar tareas más rápido y eficientemente \
que los humanos o los animales. Esta transición llevó a un aumento en la producción de \
bienes, pero también tuvo consecuencias negativas, como la explotación \laboral y la \
contaminación ambiental."""

text3 = """"El telescopio Hubble, lanzado al espacio en 1990, ha proporcionado imágenes \
impresionantes del universo y ha ayudado a los científicos a comprender mejor la cosmología."""
