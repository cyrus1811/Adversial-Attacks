import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk
from scipy import ndimage  # <-- Needed for center of mass

# Model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def denorm(batch, mean=[0.1307], std=[0.3081]):
    device = batch.device
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

class DrawingApp:
    def __init__(self, root, model, device):
        self.root = root
        self.model = model
        self.device = device
        
        self.root.title("Draw Digit & Test FGSM Attack")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        
        self.canvas_size = 400
        self.scale_factor = 14
        
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.setup_ui()
        
        self.last_x = None
        self.last_y = None
        self.drawing = False
        
    def setup_ui(self):
        title_label = tk.Label(
            self.root, 
            text="FGSM Adversarial Attack on Hand-Drawn Digits",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=10)
        
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=20)
        
        canvas_label = tk.Label(left_frame, text="Draw a digit (0-9)", font=("Arial", 12))
        canvas_label.pack()
        
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            cursor="cross"
        )
        self.canvas.pack(pady=10)
        
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=20)
        
        epsilon_label = tk.Label(right_frame, text="Epsilon Value:", font=("Arial", 11))
        epsilon_label.pack(pady=5)
        
        self.epsilon_var = tk.DoubleVar(value=0.15)
        epsilon_slider = tk.Scale(
            right_frame,
            from_=0.0,
            to=0.5,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.epsilon_var,
            length=300
        )
        epsilon_slider.pack(pady=5)
        
        button_frame = tk.Frame(right_frame)
        button_frame.pack(pady=20)
        
        predict_btn = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict_digit,
            font=("Arial", 11),
            bg="#4CAF50",
            fg="white",
            width=12,
            height=2
        )
        predict_btn.grid(row=0, column=0, padx=5)
        
        attack_btn = tk.Button(
            button_frame,
            text="FGSM Attack",
            command=self.run_attack,
            font=("Arial", 11),
            bg="#f44336",
            fg="white",
            width=12,
            height=2
        )
        attack_btn.grid(row=0, column=1, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="Clear Canvas",
            command=self.clear_canvas,
            font=("Arial", 11),
            bg="#2196F3",
            fg="white",
            width=12,
            height=2
        )
        clear_btn.grid(row=1, column=0, padx=5, pady=5)
        
        visualize_btn = tk.Button(
            button_frame,
            text="Visualize Attack",
            command=self.visualize_attack,
            font=("Arial", 11),
            bg="#FF9800",
            fg="white",
            width=12,
            height=2
        )
        visualize_btn.grid(row=1, column=1, padx=5, pady=5)
        
        results_frame = tk.LabelFrame(
            right_frame,
            text="Prediction Results",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=10
        )
        results_frame.pack(pady=20, fill=tk.BOTH)
        
        self.result_text = tk.Text(
            results_frame,
            height=15,
            width=40,
            font=("Courier", 10),
            bg="#f0f0f0"
        )
        self.result_text.pack()
        
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_on_canvas(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill="white",
                width=12,    # reduced from 20 so it's more like MNIST
                capstyle=tk.ROUND,
                smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255,
                width=12     # reduced from 20 so it's more like MNIST
            )
            self.last_x = x
            self.last_y = y
        
    def stop_drawing(self, event):
        self.drawing = False
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_text.delete(1.0, tk.END)

    # ----------- NEW PREPROCESSING FUNCTION -----------
    def preprocess_image(self):
        img_array = np.array(self.image).astype(np.float32)
        
        # If blank, return blank tensor
        if not np.any(img_array):
            img_tensor = torch.zeros(1, 1, 28, 28).to(self.device)
            return img_tensor

        # Find bounding box
        rows = np.any(img_array, axis=1)
        cols = np.any(img_array, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = img_array[rmin:rmax+1, cmin:cmax+1]

        # Scale to 20x20 (maintain aspect)
        rows_crop, cols_crop = cropped.shape
        scale = min(20.0 / rows_crop, 20.0 / cols_crop)
        new_rows = int(round(rows_crop * scale))
        new_cols = int(round(cols_crop * scale))

        cropped_img = Image.fromarray(np.clip(cropped, 0, 255).astype(np.uint8))
        resized_img = cropped_img.resize((new_cols, new_rows), Image.Resampling.LANCZOS)
        resized_array = np.array(resized_img).astype(np.float32)

        # Center of mass
        if np.sum(resized_array) > 0:
            cy, cx = ndimage.center_of_mass(resized_array)
        else:
            cy, cx = new_rows // 2, new_cols // 2

        img_28x28 = np.zeros((28, 28), dtype=np.float32)
        top = int(round(14 - cy))
        left = int(round(14 - cx))

        # Insert resized into 28x28 image
        for i in range(new_rows):
            for j in range(new_cols):
                new_i = top + i
                new_j = left + j
                if 0 <= new_i < 28 and 0 <= new_j < 28:
                    img_28x28[new_i, new_j] = resized_array[i, j]

        img_28x28 = img_28x28 / 255.0
        img_tensor = torch.from_numpy(img_28x28).unsqueeze(0).unsqueeze(0)
        transform = transforms.Normalize((0.1307,), (0.3081,))
        img_tensor = transform(img_tensor)
        return img_tensor.to(self.device)
    # ---------------- END NEW PREPROCESS -----------------

    def predict_digit(self):
        try:
            img_tensor = self.preprocess_image()
            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                prediction = output.argmax(dim=1).item()
                confidence = probabilities[prediction].item()
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=" * 35 + "\n")
            self.result_text.insert(tk.END, "ORIGINAL PREDICTION\n")
            self.result_text.insert(tk.END, "=" * 35 + "\n\n")
            self.result_text.insert(tk.END, f"Predicted Digit: {prediction}\n")
            self.result_text.insert(tk.END, f"Confidence: {confidence:.2%}\n\n")
            self.result_text.insert(tk.END, "All Probabilities:\n")
            self.result_text.insert(tk.END, "-" * 35 + "\n")
            
            for i, prob in enumerate(probabilities):
                bar = "█" * int(prob.item() * 20)
                self.result_text.insert(tk.END, f"{i}: {bar} {prob.item():.2%}\n")
                
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")
            
    def run_attack(self):
        try:
            epsilon = self.epsilon_var.get()
            img_tensor = self.preprocess_image()
            img_tensor.requires_grad = True
            
            self.model.eval()
            output = self.model(img_tensor)
            original_pred = output.argmax(dim=1).item()
            original_prob = F.softmax(output, dim=1)[0][original_pred].item()
            
            loss = F.nll_loss(output, torch.tensor([original_pred]).to(self.device))
            self.model.zero_grad()
            loss.backward()
            
            data_grad = img_tensor.grad.data
            data_denorm = denorm(img_tensor)
            perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
            perturbed_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
            adv_output = self.model(perturbed_normalized)
            adv_pred = adv_output.argmax(dim=1).item()
            adv_prob = F.softmax(adv_output, dim=1)[0][adv_pred].item()
            adv_probabilities = F.softmax(adv_output, dim=1)[0]
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=" * 35 + "\n")
            self.result_text.insert(tk.END, "FGSM ATTACK RESULTS\n")
            self.result_text.insert(tk.END, "=" * 35 + "\n\n")
            self.result_text.insert(tk.END, f"Epsilon: {epsilon}\n\n")
            
            self.result_text.insert(tk.END, "ORIGINAL:\n")
            self.result_text.insert(tk.END, f"  Prediction: {original_pred}\n")
            self.result_text.insert(tk.END, f"  Confidence: {original_prob:.2%}\n\n")
            
            self.result_text.insert(tk.END, "ADVERSARIAL:\n")
            self.result_text.insert(tk.END, f"  Prediction: {adv_pred}\n")
            self.result_text.insert(tk.END, f"  Confidence: {adv_prob:.2%}\n\n")
            
            if adv_pred != original_pred:
                self.result_text.insert(tk.END, "✓ ATTACK SUCCESSFUL!\n\n")
            else:
                self.result_text.insert(tk.END, "✗ Attack failed\n\n")
            
            self.result_text.insert(tk.END, "Adversarial Probabilities:\n")
            self.result_text.insert(tk.END, "-" * 35 + "\n")
            
            for i, prob in enumerate(adv_probabilities):
                bar = "█" * int(prob.item() * 20)
                self.result_text.insert(tk.END, f"{i}: {bar} {prob.item():.2%}\n")
            
            self.last_original = img_tensor
            self.last_adversarial = perturbed_data
            self.last_original_pred = original_pred
            self.last_adv_pred = adv_pred
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")
            
    def visualize_attack(self):
        if not hasattr(self, 'last_original'):
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please run FGSM attack first!")
            return
            
        try:
            epsilon = self.epsilon_var.get()
            original = denorm(self.last_original).squeeze().cpu().detach().numpy()
            adversarial = self.last_adversarial.squeeze().cpu().detach().numpy()
            perturbation = adversarial - original
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(original, cmap='gray')
            axes[0].set_title(f'Original\nPred: {self.last_original_pred}', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            im1 = axes[1].imshow(perturbation * 10, cmap='seismic', vmin=-1, vmax=1)
            axes[1].set_title(f'Perturbation ×10\n(ε={epsilon})', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
            axes[2].imshow(adversarial, cmap='gray')
            axes[2].set_title(f'Adversarial\nPred: {self.last_adv_pred}', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            im3 = axes[3].imshow(perturbation * 50, cmap='seismic', vmin=-1, vmax=1)
            axes[3].set_title('Difference ×50', fontsize=12, fontweight='bold')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046)
            plt.tight_layout()
            plt.savefig('drawn_digit_attack.png', dpi=300, bbox_inches='tight')
            plt.show()
            self.result_text.insert(tk.END, "\n✓ Visualization saved as 'drawn_digit_attack.png'")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Visualization Error: {str(e)}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = Net().to(device)
        model.load_state_dict(torch.load("mnist_cnn_best.pth", map_location=device))
        model.eval()
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("Error: mnist_cnn_best.pth not found!")
        print("Please run train_mnist_model.py first to train the model.")
        return

    root = tk.Tk()
    app = DrawingApp(root, model, device)
    root.mainloop()

if __name__ == '__main__':
    main()
