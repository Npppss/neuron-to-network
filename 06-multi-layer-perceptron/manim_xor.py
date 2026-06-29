from manim import *

class AnimasiMLPXOR(Scene):
    def construct(self):
        # 1. Judul Animasi
        judul = Text("Forward Pass: Memecahkan Logika XOR", font_size=32, weight=BOLD).to_edge(UP)
        self.play(Write(judul))

        # 2. Membuat Node (Neuron)
        # Input Layer: 2 Neuron (Biru)
        layer_input = VGroup(*[Circle(radius=0.3, color=BLUE, fill_opacity=0.7) for _ in range(2)])
        layer_input.arrange(DOWN, buff=1.5).shift(LEFT * 3)
        
        # Hidden Layer: 3 Neuron (Hijau)
        layer_hidden = VGroup(*[Circle(radius=0.3, color=GREEN, fill_opacity=0.7) for _ in range(3)])
        layer_hidden.arrange(DOWN, buff=1)
        
        # Output Layer: 1 Neuron (Merah)
        layer_output = VGroup(*[Circle(radius=0.3, color=RED, fill_opacity=0.7) for _ in range(1)])
        layer_output.shift(RIGHT * 3)

        self.play(Create(layer_input), Create(layer_hidden), Create(layer_output))

        # 3. Membuat Garis Koneksi (Weights)
        koneksi_in_hid = VGroup()
        for n_in in layer_input:
            for n_hid in layer_hidden:
                garis = Line(n_in.get_right(), n_hid.get_left(), stroke_width=2, stroke_opacity=0.3)
                koneksi_in_hid.add(garis)

        koneksi_hid_out = VGroup()
        for n_hid in layer_hidden:
            for n_out in layer_output:
                garis = Line(n_hid.get_right(), n_out.get_left(), stroke_width=2, stroke_opacity=0.3)
                koneksi_hid_out.add(garis)

        self.play(Create(koneksi_in_hid), Create(koneksi_hid_out))

        # 4. SIMULASI DATA MENGALIR (XOR: Input 1 dan 0)
        teks_input1 = Text("Saklar A: 1", font_size=20).next_to(layer_input[0], LEFT)
        teks_input2 = Text("Saklar B: 0", font_size=20).next_to(layer_input[1], LEFT)
        
        self.play(Write(teks_input1), Write(teks_input2))
        self.wait(0.5)

        # Animasi Aliran Data: Input -> Hidden
        self.play(koneksi_in_hid.animate.set_color(YELLOW).set_opacity(1), run_time=1)
        self.play(layer_hidden.animate.set_color(YELLOW), run_time=0.5)
        self.play(
            koneksi_in_hid.animate.set_color(WHITE).set_opacity(0.3), 
            layer_hidden.animate.set_color(GREEN), 
            run_time=0.5
        )

        # Animasi Aliran Data: Hidden -> Output
        self.play(koneksi_hid_out.animate.set_color(YELLOW).set_opacity(1), run_time=1)
        self.play(layer_output.animate.set_color(YELLOW), run_time=0.5)
        self.play(
            koneksi_hid_out.animate.set_color(WHITE).set_opacity(0.3), 
            layer_output.animate.set_color(RED), 
            run_time=0.5
        )

        # 5. Menampilkan Hasil Akhir
        teks_output = Text("Output: 1 (Lampu Nyala)", font_size=24, color=YELLOW).next_to(layer_output[0], RIGHT)
        self.play(Write(teks_output))

        self.wait(3)