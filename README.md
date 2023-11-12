<h1>Detekcja wielu sylwetek z wykorzystaniem Mediapipe</h1>
Podstawową funkcjonalnością programu jest detekcja wielu sylwetyek na strumieniu wideo z kamery lub pliku z nagraniem.

<h3>Wymagania</h3>
Aby program mógł działać potrzebane jest środowisko Python w wersji 3.10 z zainstolowaną biblioteką</br>
Mediapipe oraz Opencv. Program pozwala na wykorzystanie technologii CUDA, natomiast do poprawnego</br>
działania z tą technologią wymagane jest ręczne zbudowanie biblioteki OpenCV dla konkretnego sprzętu.</br>
</br>
Poradnik konfiguracji OpenCV dla techonologii CUDA:</br>
https://medium.com/geekculture/setup-opencv-dnn-module-with-cuda-backend-support-for-windows-7f1856691da3
</br></br>
Aplikacja wykorzystuje narzędzie YOLOv3. Do poprawnego działania narzędzia wymagane jest pobranie ze strony narzędzia</br>
plików konfiguracyjnego (.cfg) oraz z wagami sieci (.weights). Następnie umiesczenie ich w folderze "yolo" w głównym</b> 
folderze aplikacji z nazwami "yolov3-[<i>wersja np. tiny</i>].[<i>rozszerzenie weights lub cfg</i>]".  W przypadku uruchomienia programu tylko na CPU </br>
zaleca się pobranie wersji tiny a w przypadku użycia technologi CUDA wersji 320.</br>
Strona narzędzia YOLOv3:</br>
https://pjreddie.com/darknet/yolo/

<h3>Użycie</h3>
Program ma zaimplementowane dwa tryby pracy: detekcja z kamery i detekcja z pliku wideo. </br>
Wybór następuje w menu.</br></br>
<p align="center">
    <img src="https://github.com/Mcoos11/Multipose_detect/blob/main/readme_img/img_1.png" />
</p>
</br></br>

<ul>
    <li>Detekcja z kamery - do tego trybu wymagana jest podłączoina do komputera kamera np. internetowa.</li>
        W związku z możliwośią wystąpienia kilku widzianych w systemnie operacyjnym kamer program wymaga<br>
        podania nr kamery. W przypadku podłączonej tylko jednej kamery zazwyczaj jest to nr 0 lub 1.
</ul>
<ul>
    <li>Detekcja z plku wideo - do tego trybu wymagane jest umieszczenie nagrania poddawanego detekcji w folderze "test_inputs"</li>
    Nazwa pliku jest wprowadzana do programu po wyborze opcji detekcji z pliku wideo w menu.</br>
    Podczas podawania nazwy należy podać rozszerzenie pliku.
</ul>
