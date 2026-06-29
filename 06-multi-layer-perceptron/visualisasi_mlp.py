from manim import *

class JaringanSaraf(Scene):
    def construct(self):
        #Membuat Lapisan Input
        input_layer = VGroup(*[Circle(radius=0.4, color=BLUE, fill_opacity=0.5) for _ in range(2)])
        input_layer.arrange(DOWN, buff=1).shift(LEFT * 3)

        ##membuat Lapisan Tersembunyi
        hidden_layer=VGroup(*[Circle(radius=0.4, color=GREEN, fill_opacity=0.5) for _ in range(3)])
        hidden_layer.arrange(DOWN, buff=0.8)

        ## membuat Lapisan Output
        output_layer=VGroup(*[Circle(radius=0.4, color=RED, fill_opacity=0.5) for _ in range(1)])
        output_layer.shift(RIGHT * 3)

        ##animasikan muncul neuron
        self.play(Create(input_layer), Create(hidden_layer), Create(output_layer))
        self.wait(1)

        #Membuat koneksi antar neuron
        connections = VGroup()

        # Koneksi Input -> Hidden (Matriks Bobot 2x3)
        for input_node in input_layer:
            for hidden_node in hidden_layer:
                garis = Line(input_node.get_right(), hidden_node.get_left(), stroke_width=2, color=WHITE, stroke_opacity=0.5)
                connections.add(garis)

        # Koneksi Hidden -> Output (Matriks Bobot 3x1)
        # Koneksi Hidden -> Output (Matriks Bobot 3x1)
        for hidden_node in hidden_layer:
            for output_node in output_layer:
                garis = Line(hidden_node.get_right(), output_node.get_left(), stroke_width=2, color=WHITE, stroke_opacity=0.5)
                connections.add(garis)

        # Animasi menarik garis bobot secara elegan
        self.play(Create(connections), run_time=3)
        
        # Menambahkan teks judul
        judul = Text("Multi-Layer Perceptron", font_size=36).to_edge(UP)
        self.play(Write(judul))
        
        self.wait(3)