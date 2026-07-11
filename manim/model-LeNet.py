from manim import *

class LeNet5Architecture(Scene):
    def construct(self):
        # --- Judul ---
        title = Text("Arsitektur LeNet-5 (1998)", font_size=36, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # --- Fungsi Bantuan untuk Membuat Tumpukan Feature Maps ---
        def create_feature_maps(num_maps, size, offset, color):
            # Membatasi visualisasi maksimal 10 tumpukan agar tidak terlalu padat di layar
            visual_maps = min(num_maps, 10) 
            stack = VGroup(*[
                Square(side_length=size).set_stroke(color, 1.5).set_fill(color, opacity=0.1)
                for _ in range(visual_maps)
            ])
            for i, sq in enumerate(stack):
                sq.shift(RIGHT * offset * i + UP * offset * i)
            return stack

        # --- Definisi Tiap Layer ---
        # 1. Input Layer
        input_layer = Square(side_length=2.2, color=WHITE)
        in_label = Text("Input\n32x32", font_size=16).next_to(input_layer, DOWN, buff=0.5)

        # 2. C1: Convolutional Layer 1
        c1 = create_feature_maps(num_maps=6, size=1.8, offset=0.06, color=BLUE)
        c1_label = Text("C1\n6@28x28", font_size=16).next_to(c1, DOWN, buff=0.5)

        # 3. S2: Subsampling (Pooling) Layer 2
        s2 = create_feature_maps(num_maps=6, size=1.3, offset=0.06, color=RED)
        s2_label = Text("S2\n6@14x14", font_size=16).next_to(s2, DOWN, buff=0.5)

        # 4. C3: Convolutional Layer 3
        c3 = create_feature_maps(num_maps=16, size=0.9, offset=0.04, color=BLUE)
        c3_label = Text("C3\n16@10x10", font_size=16).next_to(c3, DOWN, buff=0.5)

        # 5. S4: Subsampling Layer 4
        s4 = create_feature_maps(num_maps=16, size=0.5, offset=0.04, color=RED)
        s4_label = Text("S4\n16@5x5", font_size=16).next_to(s4, DOWN, buff=0.5)

        # 6. C5/F5: Fully Connected Layer 5 (120 units)
        f5 = VGroup(*[Circle(radius=0.06, color=GREEN, fill_opacity=1) for _ in range(8)])
        f5.arrange(DOWN, buff=0.1)
        f5_label = Text("F5\n120", font_size=16).next_to(f5, DOWN, buff=0.5)

        # 7. F6: Fully Connected Layer 6 (84 units)
        f6 = VGroup(*[Circle(radius=0.06, color=GREEN, fill_opacity=1) for _ in range(6)])
        f6.arrange(DOWN, buff=0.1)
        f6_label = Text("F6\n84", font_size=16).next_to(f6, DOWN, buff=0.5)

        # 8. Output Layer (10 units)
        out = VGroup(*[Circle(radius=0.06, color=YELLOW, fill_opacity=1) for _ in range(4)])
        out.arrange(DOWN, buff=0.1)
        out_label = Text("Output\n10", font_size=16).next_to(out, DOWN, buff=0.5)

        # --- Mengatur Posisi dan Jarak ---
        layers = [input_layer, c1, s2, c3, s4, f5, f6, out]
        labels = [in_label, c1_label, s2_label, c3_label, s4_label, f5_label, f6_label, out_label]
        
        # Susun layer secara horizontal
        network_group = VGroup(*layers)
        network_group.arrange(RIGHT, buff=0.7)
        
        # Posisikan ulang label agar sejajar di bawah masing-masing layer
        for layer, label in zip(layers, labels):
            label.next_to(layer, DOWN, buff=0.4)

        # --- Membuat Garis Penghubung ---
        lines = VGroup()
        for i in range(len(layers) - 1):
            line = Line(layers[i].get_right(), layers[i+1].get_left(), color=GRAY, stroke_width=2)
            lines.add(line)

        # Tengahkan seluruh grup di layar
        full_network = VGroup(network_group, VGroup(*labels), lines)
        full_network.move_to(ORIGIN)
        full_network.shift(DOWN * 0.3) # Geser sedikit ke bawah untuk memberi ruang pada judul

        # --- Animasikan Pembuatan Jaringan ---
        self.play(Create(input_layer), Write(in_label))
        
        # Loop untuk menganimasikan transisi dari satu layer ke layer berikutnya
        for i in range(len(layers) - 1):
            self.play(
                Create(lines[i]),
                Create(layers[i+1]),
                Write(labels[i+1]),
                run_time=1.5
            )

        # Tunggu sejenak di akhir
        self.wait(3)