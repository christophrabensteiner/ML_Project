
![fh](/images/image1.png)


# The Jill-team


# Projektarbeit zu Machine Learning für Data Science 2 Lab



Magdalena Breu | 1810837687

Valentin Muhr | 1810837102

Christoph Rabensteiner | 1810837995

Jochen Paul Hollich | 1810837475

Franz Innerbichler | 1810837297




# Problembeschreibung

Es sollen Bilder von Kleidungs-Artikeln aus dem Zalando Onlineshop in 10 Klassen klassifiziert werden. 
Der Datensatz besteht aus 70000 gelabelten grayscale Bildern und kann hier heruntergeladen werden: 
[https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist). 
Die folgende Abbildung zeigt einen Ausschnitt aus dem Datensatz:

![fashion](/images/image2.png)

# Überblick Thema

Zalando Research beschäftigt sich mit Machine Learning und Deep Learning Ansätzen im Kontext von E-commerce und Fashion Retail 
wie z.B. personal size recommendation oder image recognition ([https://research.zalando.com/welcome/mission/research-projects/](https://research.zalando.com/welcome/mission/research-projects/)). 
Der Fashion-MNIST Datensatz dient als Benchmarking-Dataset für die Evaluierung von ML Algorithmen. 
Die Klassifizierung von Kleidungsartikel-Bildern findet außerdem auch Einsatz in z.B. Recommendation Engines.

[https://medium.com/tensorist/classifying-fashion-articles-using-tensorflow-fashion-mnist-f22e8a04728a](https://medium.com/tensorist/classifying-fashion-articles-using-tensorflow-fashion-mnist-f22e8a04728a)

[https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d](https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d)





Beschreibung Algorithmen

Die Bilderkennung soll mithilfe eines Convolutional Neural Networks in Tensorflow trainiert werden. Dabei soll eine simple Struktur gewählt werden mit wenigen Layern und Filtern und eine leichtgewichtige Architektur zu generieren. Das von uns entwickelte Modell soll im Folgenden auf dem Trainingsdatensatz trainiert und der Prozess von uns überwacht werden. Dabei wollen wir speziell das Verhalten von Ober- / Underfitting beobachten und wie die Netzwerkstruktur darauf Einfluss hat. Nach dem Training wird mittels des Testdatensatzes die Modellgüte und die Modell Performanz analysiert und über eine Confusion-Matrix dargestellt.

# Arbeitsschritte &amp; Aufgabenverteilung

Die folgende Tabelle zeigt die vorläufigen Arbeitsschritte und die Verteilung auf die Gruppenmitglieder:


| Aufgabe | Verantwortlich |
| --- | --- |
| Vorverarbeitung (download) der Daten | Valentin |
| Modellierung | Franz + Christoph |
| Training | Valentin + Jochen |
| Evaluierung der Modelle | Magdalena + Jochen |
| Abgabe + Bierabend | alle |

# Ziele

- Ziel ist es bis zum Projekt-Update einen lauffähiges und trainiertes Modell bereitzustellen. Dazu wollen wir unterschiedliche Modelle implementieren und die individuellen Ergebnisse vergleichen.

- Unser Ziel ist es, die Bilder den richtigen Klassen zuordnen, also zu klassifizieren. Dabei versuchen wir nicht nur besser als ein Random Klassifizierer zu sein, sondern die Accuracy bestmöglich zu maximieren. Die Accuracy wird mithilfe des Confusion Matrix dargestellt.
- Das ideale Endergebnis unseres Projektes besteht in erster Linie darin, dass wir Studierenden an diesem Projekt etwas erlernen und entsprechend Erfahrungen auf dem Gebiet des Machine Learnings sammeln.