# Kippt die Entscheidung? – Schwelle, Tornado und eine geschätzte Wahrscheinlichkeit – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kipp-schwelle-demo.streamlit.app/)** -->

Eine von mehreren Demos zur **Entscheidungsanalyse** (Entscheiden unter Unsicherheit) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie nimmt die Annahme heraus, die
[`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo) noch macht: dass die Wahrscheinlichkeit, mit der gerechnet wird, **bekannt** ist. Hier wird sie aus n Beobachtungen
**geschätzt**, und die Demo zählt, wie oft die Entscheidung dann am anderen Ufer der Schwelle läge. Dazu kommen zwei Fragen davor: **wo** die Wahl bei jeder Eingabe kippt (Schwelle und Kipp-Entfernung) und
**was ein Tornado-Diagramm** davon zeigt.

**Das Zahlenbeispiel ist ausgedacht** (Kapazitätsfall, Geldeinheiten GE) und ein **Modell, kein Fall aus der Praxis**: G (groß bauen) zahlt +200 GE bei hoher und −100 GE bei niedriger Nachfrage, K (klein bauen
oder mieten) sicher +30 GE, p = 1/2 ist die Wahrscheinlichkeit hoher Nachfrage; bewertet wird der Erwartungswert (G 50 GE, K 30 GE, Vorsprung von G 20 GE). Die n Beobachtungen sind **gedachte frühere Perioden
mit bekanntem Ausgang** (eine Annahme des Modells, kein Datensatz). Nachbarn: [`baum-strategie-demo`](https://github.com/sebastian-hanisch/baum-strategie-demo) (Erwartungswert und Strategie im Baum),
[`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo) (Vorab und Trefferquote sind dort gegeben), [`risiko-aequivalent-demo`](https://github.com/sebastian-hanisch/risiko-aequivalent-demo)
(die Wahl kippt dort an der Risikotoleranz statt an der Wahrscheinlichkeit), [`stichprobenbaum-demo`](https://github.com/sebastian-hanisch/stichprobenbaum-demo) (Fehlerquote über Wiederholungen bei n gezogenen Szenarien),
[`nutzen-erheben-demo`](https://github.com/sebastian-hanisch/nutzen-erheben-demo) (Rauschen in der Nutzenerhebung als Quelle des Wackelns).

## Warum dieses Problem

Wer eine Wahrscheinlichkeit aus wenigen Beobachtungen schätzt und dann rechnet, als wäre sie bekannt, übersieht, dass der Entschluss am anderen Ufer der Schwelle liegen könnte. **Wo kippt die Entscheidung, was sagt ein
Tornado-Diagramm darüber, und wie oft läge die andere Alternative vorn, wenn p nur geschätzt ist?**

Im Kapazitätsfall (G +200/−100 GE, K +30 GE, p = 1/2) kippt die Wahl bei p* = 13/30 = 0,433; relativ zum Basiswert sind das −13,3 % für p, −20 % für G, +40 % für L und +66,7 % für K. Ist p aus n = 20
Beobachtungen mit 10 Perioden hoher Nachfrage geschätzt (gleichverteiltes Vorab), läge in 26,7 % der Posterior-Ziehungen die andere Alternative vorn; unter 5 % erst ab n = 150 (gerade n, Anteil 1/2, gleichverteiltes
Vorab), unter 1 % ab n = 302. Mit dem engen Vorab Beta(20, 20) sind es 15,1 %: die Zahl hängt am Vorab. Das Tornado-Diagramm sagt etwas anderes als das Kippen: seine Reihenfolge ist Konvention (bei „alle ±20 %“
steht p auf Rang 1, bei p ±20 %, G ±50 %, L ±30 %, K ±5 % steht G auf Rang 1; bei zufällig gezogenen Spannweiten wechselt Rang 1 in 46,5 % und 46,1 % von je rund 147 400 Zufallsproblemen), und die
Kipp-Entfernung ordnet bei gleichen relativen Spannweiten umgekehrt zur Tornado-Breite (Δ linear in jeder Eingabe, p ohne Kappung bei 1), ist also kein unabhängiges zweites Maß. Bei in p linearer Auszahlung entscheidet nur das Posterior-Mittel; der Kippanteil ist die
Posterior-Wahrscheinlichkeit, dass die andere Alternative besser wäre, **kein Grund zu wechseln**, und sein Preis ist der Wert der Information über p (8,70 GE bei n = 10, 0,60 GE bei n = 100).
Alles gilt in diesem Modell, bei diesen Parametern; die Zähleinheit steht je Aussage dabei (Ziehung, Wiederholung, Zufallsproblem, Eingabe).

## Modell

* **Vorsprung:** Δ(p) = p·G − (1 − p)·L − K = (G + L)·(p − p*), linear in p und in jeder anderen Eingabe. **Schwelle** p* = (L + K)/(G + L); in Kostenschreibweise ist r = (G − K)/(L + K) das Verhältnis von verpasster
  Chance (K gewählt, Nachfrage hoch) zu unnötigem Bau (G gewählt, Nachfrage niedrig) und p* = 1/(1 + r).
* **Kipp-Entfernung** einer Eingabe: ihr Abstand zu dem Wert, bei dem Δ = 0 wird, geteilt durch den Basiswert (relativ; die Wahl „relativ“ ist selbst eine Konvention). **Gleichstand:** |Δ| ≤ 1e-9 (in Brüchen exakt, in
  Gleitkomma bleibt ein Rest); bei Gleichstand gilt G nicht als gekippt. **Tornado:** Breite einer Eingabe = |Δ(oben) − Δ(unten)| bei Variation nur dieser Eingabe um ihre halbe Spannweite (Prozent des Basiswerts, p nach oben bei 1
  gekappt), alle anderen auf dem Basiswert; „kippt“ heißt, dass das Vorzeichen von Δ strikt wechselt.
* **Geschätztes p:** n Beobachtungen, k mal hohe Nachfrage; Vorab Beta(a₀, b₀) (gleichverteilt Beta(1, 1), Jeffreys Beta(½, ½) oder eng Beta(20, 20), so viel Gewicht wie 40 frühere Beobachtungen); Posterior Beta(k + a₀, n − k + b₀).
* **Kippanteil** (Zähleinheit **Ziehung**): Anteil der Posterior-Ziehungen, in denen die vom Posterior-Mittel gewählte Alternative nicht die bessere ist (bei gewähltem G: P(p < p*)). **Fehlerquote** (Zähleinheit
  **Wiederholung**): Anteil der Wiederholungen der Erhebung bei festem wahrem p, in denen die Regel „G, wenn das Posterior-Mittel strikt über p* liegt“ die schlechtere Alternative wählt. Beides sind verschiedene Größen.
* **Verlust** = Wert der vollkommenen Information über p: E[max Δ] − max(E[Δ], 0) in GE, also die ex-post-Mehrkosten gegenüber der besseren Alternative, gemittelt über die Posterior-Ziehungen (im Bestand der Demos heißt das
  „Regret“; gemeint sind überall ex-post-Mehrkosten gegen ein Orakel, nicht das Savage-Kriterium). Er ist nicht der Wert der vollkommenen Information über den **Zustand** (hohe oder niedrige Nachfrage): der ist bei p = 1/2 gleich 65 GE.
* **Nenner:** Werte stehen absolut in GE; wo ein Anteil steht, steht sein Nenner dabei (Ziehungen, Wiederholungen, Zufallsprobleme, Basiswert).
* **Drei Zustände:** ein zweites ausgedachtes Beispiel (G zahlt 250 / 60 / −150 GE, K zahlt 90 / 50 / 0 GE, Anteile 0,3 / 0,4 / 0,3, Erwartungswert 54 gegen 47 GE, Vorsprung 7 GE) mit Dirichlet-Posterior; Δ = 310 π₁ + 160 π₂ − 150.

## Methodik

* **Exakt:** Kippanteil über `scipy.special.betainc`, bei ganzzahligen Parametern zusätzlich als Binomialsumme in Brüchen; Verlust aus derselben Verteilungsfunktion (Formel in der App); Fehlerquote über `scipy.stats.binom`; drei Zustände durch
  eine eindimensionale Quadratur (die Dichte von π₁, innen die Betaverteilung des Anteils von π₂ am Rest).
* **Ziehungen** allein aus dem ganzzahligen SplitMix64 (`kps_rng.py`, geprüfte Vorlage, unverändert kopiert; Zufalls-Seed als Zahl): Beta(a, b) mit ganzzahligen Parametern als a-kleinste von a + b − 1 Gleichverteilten, Dirichlet als
  zusammengefasste Abstände sortierter Gleichverteilter (`kps_draws.py`). Das braucht weder Logarithmus noch Kosinus der C-Bibliothek. Eine Ziehung kostet a + b − 1 Gleichverteilte; eine Obergrenze von 4 Millionen je Rechnung
  begrenzt die Ziehungen bei großem n (n = 1 000: höchstens 3 996 Ziehungen). Beim Jeffreys-Vorab sind die Parameter nicht ganzzahlig: dort kommen die Ziehungen aus der Umkehrfunktion der Betaverteilung (`scipy.special.betaincinv`),
  angewandt auf Gleichverteilte desselben Stroms.
* **Zufallsprobleme** (Zähleinheit Zufallsproblem): p gleichverteilt in [0,2; 0,8], G in [100; 300], L in [50; 150], K gleichverteilt zwischen 0 und dem Erwartungswert von G, nur Probleme mit mehr als 5 GE Erwartungswert von G
  (G gewählt); das ist eine **Annahme dieser Messung**, keine allgemeine Verteilung von Entscheidungsproblemen. Zufällige Spannweiten: halbe Spannweite je Eingabe gleichverteilt zwischen 5 und 50 %.
* **Tornado als Formel:** Δ ist linear, Breite = |Ableitung| × Spannweite; Gegenprobe im Orakel-Test durch Auswerten von Δ an den Enden der Spannweite je Problem.
* Alles rechnet live (Rechenzeit unten); nichts ist vorgerechnet oder eingefroren.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten im Kapazitätsfall mit den genannten Einstellungen; die Spalte „Test“ nennt den Anfang des Testnamens, der die Zahl rechnet (`tests/test_claims.py`, `test_b1…` bis `test_b17…`; Zählungen und Anteile als Band, Brüche und Gleitkommawerte aus exakter Rechnung
mit relativer Toleranz 1e-9).

| Nr | Befund (Voraussetzung) | Test |
|---|---|---|
| B1 | Schwelle p* = 13/30 = 0,433; Schwellen G* = 160, L* = 140, K* = 50 (jeweils die anderen Eingaben auf dem Basiswert p = 1/2, G = 200, L = 100, K = 30); relative Kipp-Entfernung p 13,3 %, G 20 %, L 40 %, K 66,7 %; Kostenverhältnis r = 17/13 | `test_b1` |
| B2 | Tornado bei „alle ±20 %“: Breiten p 60, G 40, L 20, K 12 GE (p kippt, G liegt an der unteren Grenze genau im Gleichstand, kippt nicht), Rang 1 ist p. Bei p ±20 %, G ±50 %, L ±30 %, K ±5 %: Breiten p 60, G 100, L 30, K 3 GE (p und G kippen), Rang 1 ist G | `test_b2` |
| B3 | Breite × relative Kipp-Entfernung ist für jede Eingabe gleich 2x·Vorsprung (8 GE bei x = 20 %); jede Konvention „alle ±x %“ (x = 5, 10, 20, 50 %) gibt dieselbe Reihenfolge p > G > L > K | `test_b3` |
| B4 | Zufallsproblem-Zählung (200 000 Zufallsprobleme je Zufalls-Seed, davon 147 366 bei Zufalls-Seed 1301 und 147 367 bei 1302 mit G gewählt): bei zwei unabhängig gezogenen Konventionen (halbe Spannweite 5 bis 50 %) wechselt Rang 1 in 46,5 % und 46,1 % der Probleme, bleibt also in 53,5 % und 53,9 % gleich; die ganze Reihenfolge wechselt in 74,8 % und 74,8 %. Gegen die Kipp-Entfernung (alle Spannweiten gleich) wechselt Rang 1 in 33,3 % und 33,2 %, die ganze Reihenfolge in 60,1 % und 60,4 % | `test_b4_b5` |
| B5 | „alle ±20 %“ gegen die Kipp-Entfernung: in 0 von 147 366 und 0 von 147 367 Zufallsproblemen weicht die Reihenfolge ab; Rang 1 der Kipp-Entfernung ist p in 147 366 von 147 366 und 147 367 von 147 367 | `test_b4_b5` |
| B6 | „p hat die kleinste relative Kipp-Entfernung“ gilt bei gewähltem G, G > 0, L ≥ 0, K ≥ 0 in 23 640 von 23 640 Zufallsproblemen (Beweis in drei Zeilen, Orakel-Test); ohne die Bedingung (G, L, K beliebigen Vorzeichens, beide Vorzeichen des Vorsprungs, 400 000 Zufallsprobleme, Zufalls-Seed 1310) nur in 22,2 %, bei Vorsprung über 0 in 41,2 % (56 726 von 137 785), bei Vorsprung unter 0 in 12,2 % (31 903 von 262 215) | `test_b6` |
| B7 | Kippanteil exakt (p̂ = 1/2, k = n/2, gleichverteiltes Vorab): n = 10: 32,48 %, 20: 26,73 %, 40: 19,41 %, 100: 8,86 %, 200: 2,86 %, 400: 0,36 %, 1 000: 0,001 %; Binomialsumme in Brüchen bei n = 10: 26 634 858 184 181 / 82 012 500 000 000 | `test_b7` |
| B8 | Größtes gerades n mit Kippanteil über 5 %: 148 (unter 5 % ab n = 150); über 1 %: 300 (unter 1 % ab n = 302); Raster n = 2, 4, …, 20 000, k = n/2, gleichverteiltes Vorab | `test_b8` |
| B9 | Vorab (n = 20 / n = 100, k = n/2): gleichverteilt 26,73 % / 8,86 %, Jeffreys 27,21 % / 8,97 %, eng Beta(20, 20) 15,08 % / 5,68 % | `test_b9` |
| B10 | Verlust (Wert der Information über p), exakt, k = n/2, gleichverteilt: n = 10: 8,70 GE, 20: 5,01 GE, 40: 2,42 GE, 100: 0,60 GE, 200: 0,11 GE, 400: 0,008 GE; zum Vergleich der Vorsprung 20 GE und der Wert der Information über den Zustand 65 GE | `test_b10` |
| B11 | Näherung Φ(−z) mit p̂ = k/n: 33,66 % bei n = 10 gegen exakt 32,48 %; Abweichung 0,26 Prozentpunkte bei n = 100, danach kleiner (0,10 / 0,02 / unter 0,01 bei n = 200 / 400 / 1 000) (Näherung, kein Orakel) | `test_b11` |
| B12 | Linearität: der Erwartungswert des Vorsprungs über den Posterior ist der Vorsprung beim Posterior-Mittel (exakt in Brüchen); bei G2 (+220 GE bei zwei hohen Perioden mit Wahrscheinlichkeit p², sonst −100 GE, K = 30 GE, gleichverteilt) entscheiden Einsetzen und Mitteln in 19 von 5 141 Datensätzen (n = 4 bis 100, alle k) verschieden; kleinster Fall n = 6, k = 4: Einsetzen −5 GE, Mitteln 10/3 GE | `test_b12` |
| B13 | Fehlerquote über Wiederholungen (exakt binomial, Regel „G, wenn das Posterior-Mittel strikt über 13/30 liegt“, gleichverteilt): wahres p = 1/2 (Vorsprung 20 GE): n = 10: 37,70 % (193/512, Verlust 7,54 GE je Wiederholung), 20: 25,17 %, 40: 21,48 %, 100: 9,67 %, 200: 2,80 %, 400: 0,40 %; wahres p = 0,45 (Vorsprung 5 GE): 50,44 % / 41,43 % / 43,91 % / 38,28 % / 31,01 % / 25,71 % und bei n = 1 000 noch 14,71 % (Verlust 0,74 GE); wahres p = 0,40 (Vorsprung von K 10 GE) bei n = 1 000: 1,56 % | `test_b13` |
| B14 | Die Fehlerquote ist nicht monoton in n: bei wahrem p = 0,45 gilt 50,44 % (n = 10), 41,43 % (n = 20), 43,91 % (n = 40); bei wahrem p = 1/2 steigt sie zwischen n = 10 und n = 600 in 255 von 590 Schritten von n auf n + 1 (Deutung Ganzzahligkeit von k, nicht geprüft) | `test_b14` |
| B15 | Drei Zustände, Kippanteil durch Quadratur (Anteile 0,3 / 0,4 / 0,3, gleichverteiltes Vorab): ohne Daten 225/496 = 45,36 % der Fläche des Dreiecks (exakt), N = 10: 41,51 %, 30: 36,80 %, 100: 27,68 %, 300: 15,51 %, 1 000: 3,25 %; Verlust 22,68 / 9,88 / 5,26 / 2,02 / 0,56 / 0,048 GE (N = 0 / 10 / 30 / 100 / 300 / 1 000) | `test_b15` |
| B16 | Ziehungen gegen exakt (Band 4,5 Standardfehler): Beta(11, 11) bei Zufalls-Seed 1301 und 100 000 Ziehungen 26,68 % gegen exakt 26,73 %; Dirichlet bei N = 30 und N = 10 jeweils im Band; bei n = 1 000 sind es 3 996 Ziehungen | `test_b16` |
| B17 | Regressionsanker (kein Literaturwert): für die Investition +100/−60/+5 (Schwelle 13/32) mit 9 hohen Perioden von 20 ergab die erste Rechnung 0,330; diese Demo rechnet 0,3298 | `test_b17` |

### Zahl → Test

Die Tests in `tests/test_claims.py` rechnen jede Zahl der Tabelle und suchen ihre Schreibweise in dieser README. Exakt (Brüche, Zählung kleiner Mengen): B1, B2, B3, die Bruchzahlen von B7, B12, B13 (193/512), B15 (225/496).
Band: Zufallsproblem-Zählungen (± 2 Probleme), Ziehungen (4,5 Standardfehler), alle scipy-Werte (relativ 1e-9 gegen die Bruchrechnung). Nicht im Test: die Rechenzeit (einmal gemessen, siehe „Ehrliche Grenzen“).

## Befunde und Korrekturen gegenüber der ersten Rechnung

Eine frühere Messung (anderer Zufallsstrom, eine Investition statt des Kapazitätsfalls für die Vorgängerzahlen) diente als Erwartung; die Zahlen dieser Demo gelten, wo sie abweichen:

* **Zufallsproblem-Zählungen:** 46,5 % und 46,1 % (SplitMix64) statt 46,6 % und 46,8 % (numpy-Strom): dieselbe Verteilung, anderer Strom, Unterschied innerhalb der Streuung zweier Zufalls-Seeds. Die Zahl der Probleme mit G gewählt ist 147 366 und 147 367.
* **Zweite Konvention:** die erste Rechnung ließ p um ±0,10 (absolut) variieren, diese Demo um ±20 % (relativ; bei p = 1/2 dasselbe). Rang 1 unter „p ±20 %, G ±50 %, L ±30 %, K ±5 %“ gegen „alle ±20 %“ wechselt dadurch in allen Zufallsproblemen
  (100 %) statt in 99,0 %; das liegt an dieser Verteilung der Probleme (p zwischen 0,2 und 0,8, G zwischen 100 und 300, L zwischen 50 und 150) und ist keine allgemeine Häufigkeit.
* **Bedingung für „p am nächsten“:** 23 640 von 23 640 statt 23 621 von 23 621 (anderer Strom). Ist K gewählt, gilt die Aussage nicht (12,2 %); die App sagt das ausdrücklich.
* **Kippanteil bei gewähltem K:** die erste Rechnung zählte nur P(p < p*), also nur den Fall G gewählt. Diese Demo zählt allgemein die Ziehungen, in denen die vom Posterior-Mittel gewählte Alternative nicht die bessere ist. Folge, im
  Eigenschaftstest gefunden: der Kippanteil fällt in n bei festem Anteil über der Schwelle **nur**, solange G gewählt bleibt; wechselt die Wahl (enges Vorab, Mittel unter der Schwelle), springt er (Beispiel Beta(20, 20), Anteil 0,6, Schwelle 0,533: bei n = 10 wählt das Mittel K, bei n = 20 liegt es genau auf der Schwelle, bei n = 30 wählt es G).
* **Verlust bei drei Zuständen:** exakt (Quadratur) 9,88 GE bei N = 10 statt 9,92 GE aus Ziehungen; die Ziehungen dieser Demo geben 9,87 GE bei 100 000 Ziehungen, im Band der Standardfehler.
* **Ziehungen:** bei großem n sinkt die Zahl der Ziehungen (Obergrenze 4 Millionen Gleichverteilte); die erste Rechnung zog 1 Million Ziehungen mit numpy. Beim Jeffreys-Vorab Umkehrfunktion statt Ordnungsstatistik.
* **Voreinstellung N = 100** in Experiment 5 (nicht 30), damit das Preset „Drei Zustände, N = 30“ eine andere Einstellung ist als der Standard.

## Quellen

Nur als Hinweise, **nicht im Wortlaut gelesen**; alle Formeln und Zahlen dieser Demo sind eigene Rechnung in diesem Modell:

* Schwellenwahrscheinlichkeit, Beta-Posterior, Tornado: Lehrbuchstoff. Clemen/Reilly, *Making Hard Decisions*, 3. Aufl. 2013 (nur Kapitelliste gesehen) und Morgan/Henrion, *Uncertainty*, 1990, Kapitel 8 (nur Katalogeintrag gesehen).
* Briggs u. a. 2012, Value in Health 15(5), 835–842 (nur das Abstract gesehen): probabilistische Sensitivitätsanalyse mit dem Wert der vollkommenen Information als Darstellung.
* Felli/Hazen 1998, Medical Decision Making 18(1), 95–109 (nur eine Zusammenfassung gesehen): die Wahrscheinlichkeit eines Entscheidungswechsels und die Auszahlungsänderung sind verschiedene Größen, der Wert der vollkommenen Information verbindet beide.
* Grenzen des Tornado-Diagramms (Spanne und Unabhängigkeit; nur Zusammenfassungen gesehen): als Hinweis, nicht als belegte Aussage.

## Abgrenzung zu Nachbar-Demos

* [`newsvendor-evpi-demo`](https://github.com/sebastian-hanisch/newsvendor-evpi-demo) und [`saa-demo`](https://github.com/sebastian-hanisch/saa-demo): dort sind die Szenarien die Verteilung, beziehungsweise eine Stichprobe daraus für einen Plan (lineares
  Programm); hier gibt es eine einzige Wahrscheinlichkeit, einmal geschätzt, eine Ja-Nein-Entscheidung, exakt in Brüchen und Beta-Verteilung. Der Gewinner-Fluch (Optimizer's Curse) ist dort und in
  [`wen-behandeln-demo`](https://github.com/sebastian-hanisch/wen-behandeln-demo) und [`multiple-testing-demo`](https://github.com/sebastian-hanisch/multiple-testing-demo) gemessen und hier nur ein Verweis.
* [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo): dieselbe Schwellenformel 1/(1 + r), aber das p̂ kommt dort aus einem Modell je Container und die Frage ist, ob p̂ als Häufigkeit stimmt;
  hier eine Entscheidung und die Unsicherheit der Schätzung. Kalibrierung wird hier nicht gezeigt.
* [`wen-behandeln-demo`](https://github.com/sebastian-hanisch/wen-behandeln-demo): „Kipppunkt“ meint dort den Pilotumfang, ab dem die gezielte Auswahl besser ist als die beste feste Regel. Hier heißen die Größen Schwelle, Kipp-Entfernung und Kippanteil.
* [`sensitivitaet-demo`](https://github.com/sebastian-hanisch/sensitivitaet-demo): „Sensitivität“ meint dort die einer Kausalschätzung gegen eine verborgene Ursache; hier die **Sensitivität der Entscheidung** gegen ihre Eingaben.
* Die Rangakzeptanz bei SMAA (Mehrkriterien-Ranking) zieht aus dem Simplex der Kriteriengewichte, rechnet einen linearen Wert und zählt; nach der Modellform ist das dasselbe Ziehverfahren wie in Experiment 5 (dort Gewichte, hier
  Wahrscheinlichkeiten). Die Demo dazu ist nicht gelesen, die Gleichheit nicht nachgerechnet.
* Das Wort „Entscheidungsbaum“ steht in dieser Demo nicht für den Klassifikationsbaum aus Daten.

## Ehrliche Grenzen

* **Ausgedachtes Zahlenbeispiel:** G, L, K und alle Auszahlungen sind erfunden, die Beobachtungen sind gedachte frühere Perioden. Kein Befund über einen echten Betrieb.
* **Das Vorab ist eine Wahl:** dieselben Daten (n = 20, k = 10) geben 26,73 %, 27,21 % oder 15,08 %, je nach Vorab. Die Zahl bedeutet „Häufigkeit in Welten, in denen p aus dem gewählten Vorab stammt“ (Kalibrierungstest mit Zufallswelten:
  in einer Welt mit p gleichverteilt stimmt der Posterior, in einer Welt mit p ~ Beta(20, 20) und gleichverteiltem Vorab überschätzt er den Kippanteil), nichts darüber hinaus.
* **Der Kippanteil ist kein Wechselgrund:** bei in p linearer Auszahlung entscheidet das Posterior-Mittel; der Kippanteil ist die Posterior-Wahrscheinlichkeit der anderen Alternative, nicht die Fehlerquote über Wiederholungen
  (zwei Zähleinheiten, im Experiment 4 nebeneinander).
* **Nur lineare Auszahlung in p ist gemessen:** die Aussage „das Mittel genügt“ gilt dafür; bei G2 entscheiden Einsetzen und Mitteln verschieden. Ob das Posterior-Mittel auch bei Nutzenfunktionen genügt, ist **nicht gemessen**.
* **Tornado:** gemeinsames Kippen mehrerer Eingaben und korrelierte Eingaben sind **nicht gemessen**; der Rang des Tornado gegen den Rang nach dem Wert der Information je Eingabe (EVPPI) ist **nicht gemessen**, weil dafür jede Eingabe eine
  Verteilung bräuchte und jede Zahl eine neue Annahme wäre.
* **Zufallsproblem-Zählungen** gelten für die genannte Verteilung der Probleme; „Rang 1 wechselt in 46,5 %“ ist keine allgemeine Häufigkeit, und in 53,5 % bleibt er gleich: die Konvention ändert Rang 1 nicht immer, aber oft.
* **Wiederholungen:** die Ursache des Sägezahns der Fehlerquote in n (Ganzzahligkeit von k) ist eine Deutung, nicht geprüft; bei Vorab und Schwelle, die die Wahl wechseln lassen, ist die Aussage „fällt in n“ falsch (siehe oben).
* **Der Wert der Information über zusätzliche Beobachtungen** (nicht der vollkommenen Information über p) wird hier nicht gerechnet.
* **Plattform:** Zählungen und Ziehungen sind Zählungen aus einem ganzzahligen Zufallsstrom und auf jeder Plattform dieselben, bis auf die Reihenfolge der Gleitkomma-Summen; die Tests prüfen sie als Band. Unter Windows entwickelt; der Linux-Lauf (Docker) steht unter „Verifikation“.
* **Rechenzeit:** gemessen unter Windows, einmal: jede der fünf Rechnungen dauert bei den Voreinstellungen höchstens etwa 0,2 s, mit allen aufwendigen Reglern am Maximum (n = 1 000, 200 000 Ziehungen, 200 000 Zufallsprobleme, N = 1 000, enges Vorab, G2) zusammen etwa 0,3 s (nicht im Test; ein Test prüft nur, dass der Lauf mit allen Reglern am Maximum ohne Fehler endet).

## Verifikation

* **Handwerte in Brüchen:** Schwellen, Kipp-Entfernungen, Tornado-Breiten, Kippanteil bei n = 10 als Bruch, Verlust, 193/512 und 965/128, 225/496, Einsetzen gegen Mitteln (`tests/test_model.py`, `tests/test_posterior.py`).
* **Orakel-Tests** (`tests/test_oracle_kipp_schwelle.py`): Kippanteil über `scipy.special.betainc` gegen Binomialsumme und Polynomintegration in Brüchen gegen Quadratur; Verlust beide Richtungen gegen Quadratur; drei Zustände eindimensional gegen
  `dblquad`; Fehlerquote in Brüchen gegen `scipy.stats.binom` gegen Ziehen von k mit numpy (Band); Zählungen über Zufallsprobleme gegen die Einzelrechnung je Problem; Beweis „p am nächsten“ in Brüchen mit Gegenprobe; Kalibrierung mit
  Zufallswelten; Zählung Einsetzen gegen Mitteln gegen die Betafunktion; Regressionsanker.
* **Eigenschaftstests** (`tests/test_eigenschaften_kipp_schwelle.py`): nur k zählt; Skalieren und Verschieben der Auszahlungen; Vertauschen von hoch und niedrig; Kippanteil und Verlust fallen in n im Raster (nur wo G gewählt ist); Erwartungswert des
  Vorsprungs; alle „alle ±x“ geben dieselbe Reihenfolge; ohne Daten ist der Kippanteil die Schwelle.
* **Verdikte** als reine Funktionen (`kps_evaluation.verdict_*`), jede Zahl und jede Zweigart an handgesetzten Werten (`tests/test_verdicts.py`); je Preset die erwartete Wirkung (`tests/test_presets.py`).
* **Randfälle:** Gleichstand (in Gleitkomma mit Rest), k = 0 und k = n, n = 0, Schwelle außerhalb von [0, 1], Basiswert 0, Kappung von p bei 1, Dirichlet mit Eintrag 0, alle Regler am Rand (`tests/test_edge_cases.py`).
* **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, alle Vorab- und Auszahlungsauswahlen, Permalink, alle Regler am Maximum zugleich), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives
  Preset hervorgehoben; jede Änderung eines Reglers, auch im Hauptbereich, hebt die Markierung auf), Footer, Laufzeit-Importe.
* **Testzahl:** 216 Tests (`pytest --collect-only -q tests`).
* **Linux-Lauf (Docker, Python 3.12, frische Installation der ungepinnten Pakete):** alle Tests grün, einschließlich der Zählungen über Zufallsproblemen als Band (± 2).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kernfragen, fünf Experimente, Grenzen, Formeln |
| `kps_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kps_draws.py` | vektorisierter SplitMix64-Strom, Ziehungen aus Beta und Dirichlet über Ordnungsstatistiken |
| `kps_evaluation.py` | Analysen der Experimente, Zählungen, Verdikte als reine Funktionen |
| `kps_layout.py` | Fester Kopf (Stylesheet) |
| `kps_model.py` | Kapazitätsfall: Vorsprung, Schwellen, Kipp-Entfernung, Tornado, Zählungen über Zufallsprobleme |
| `kps_posterior.py` | Beta- und Dirichlet-Posterior, Kippanteil, Verlust, Fehlerquote über Wiederholungen, Einsetzen gegen Mitteln |
| `kps_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kps_rng.py` | Zufallsgenerator SplitMix64 (geprüfte Vorlage, unverändert kopiert) |
| `kps_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, plotly |
| `tests/` | Handwerte, Orakel, Eigenschaften, Claims, Verdikte, Randfälle, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Zufallsstrom |

## Bewusst nicht umgesetzt

* **Wert zusätzlicher Beobachtungen** (Beta-Binomial-Vorabanalyse) und der **Test mit Trefferquote**: gehören zur Nachbar-Demo [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo).
* **Tornado gegen Wert der Information je Eingabe, gemeinsames Kippen, korrelierte Eingaben:** würden für jede Eingabe eine Verteilung verlangen; jede Zahl wäre eine neue Annahme.
* **Nutzenfunktionen:** nur Erwartungswert; die Risikoeinstellung ist Thema von [`risiko-aequivalent-demo`](https://github.com/sebastian-hanisch/risiko-aequivalent-demo).
* **Kalibrierung eines Modells:** siehe `kalibrierung-schwelle-demo`.
* **PDF-Export:** nicht vorhanden.
* **Mehr als drei Zustände** und andere Auszahlungen im Dirichlet-Fall: Experiment 5 ist ein festes zweites Beispiel.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html).
