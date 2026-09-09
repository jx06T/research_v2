import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

def plot_gan_results(source_imgs, target_imgs, fake_imgs, labels, epoch, config, title_suffix):
    source_imgs = source_imgs.detach().cpu()
    target_imgs = target_imgs.detach().cpu()
    fake_imgs = fake_imgs.detach().cpu()

    n_samples = source_imgs.size(0)
    if n_samples == 1:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes = axes[np.newaxis, :]
    else:
        fig, axes = plt.subplots(3, n_samples, figsize=(2 * n_samples, 6))

    main_title = (f"{title_suffix} | Epoch {epoch}/{config.num_epochs}\n"
                  f"n={config.bottleneck_size}, L1_w={config.lambda_l1}\n"
                  f"lr={config.lr}, bs={config.batch_size}")

    # 使用我們自己的中文字體，解決 Matplotlib 顯示中文字元變方塊的問題
    font_prop = fm.FontProperties(fname="data/fonts/NotoSansTC-Regular.ttf")

    for j in range(n_samples):
        lbl = labels[j].item() if isinstance(labels, torch.Tensor) else labels[j]

        ax_src = axes[0, j] if n_samples > 1 else axes[0]
        ax_src.imshow(source_imgs[j].squeeze(), cmap="gray")
        ax_src.set_title(f"Source ({lbl})", fontsize=8, fontproperties=font_prop)
        ax_src.axis("off")

        ax_tgt = axes[1, j] if n_samples > 1 else axes[1]
        ax_tgt.imshow(target_imgs[j].squeeze(), cmap="gray")
        ax_tgt.set_title(f"Target ({lbl})", fontsize=8, fontproperties=font_prop)
        ax_tgt.axis("off")

        ax_gen = axes[2, j] if n_samples > 1 else axes[2]
        ax_gen.imshow(fake_imgs[j].squeeze(), cmap="gray")
        ax_gen.set_title("Generated", fontsize=8, fontproperties=font_prop)
        ax_gen.axis("off")

    plt.suptitle(main_title, fontproperties=font_prop)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.show()

def plot_training_losses(history, config):
    plt.figure(figsize=(10, 5))
    
    def smooth(scalars, weight=0.85):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    if "D_loss" in history and len(history["D_loss"]) > 0:
        plt.plot(history["D_loss"], label="Discriminator Loss (Raw)", alpha=0.3, color='blue')
        plt.plot(smooth(history["D_loss"]), label="D Loss (Smooth)", alpha=0.9, color='blue')
        
    if "G_loss" in history and len(history["G_loss"]) > 0:
        plt.plot(history["G_loss"], label="Generator Loss (Raw)", alpha=0.3, color='orange')
        plt.plot(smooth(history["G_loss"]), label="G Loss (Smooth)", alpha=0.9, color='orange')
        
    if "L1_loss" in history and len(history["L1_loss"]) > 0:
        plt.plot(smooth(history["L1_loss"]), label="L1 Pixel (Smooth)", linestyle="--", alpha=0.7, color='green')

    plt.xlabel("Iterations (logged step)")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training Losses (n={config.bottleneck_size}, L1 ratio={config.lambda_l1})")
    plt.grid(True, alpha=0.3)
    plt.show()

def preprocess_char(dataset, char_str):
    rendered = dataset._render_char_to_tensor(char_str, dataset.font_a)
    img_tensor = 1.0 - rendered
    if not dataset.invert:
        img_tensor = 1.0 - img_tensor
    return img_tensor

def test_specific_chars(G, dataset, config, epoch, chars_to_test=None, device="cpu"):
    if chars_to_test is None:
        chars_to_test = ["t","R", "G", "7","田","名","朝","明", "永", "灣", "體"]

    source_list = []
    target_list = []

    for char in chars_to_test:
        img_a = preprocess_char(dataset, char)
        source_list.append(img_a)

        raw_b = dataset._render_char_to_tensor(char, dataset.font_b)
        img_b = 1.0 - raw_b
        if not dataset.invert:
            img_b = 1.0 - img_b
        target_list.append(img_b)

    src_batch = torch.stack(source_list).to(device)
    tgt_batch = torch.stack(target_list).to(device)

    with torch.no_grad():
        z = torch.randn(len(chars_to_test), config.nz, config.bottleneck_size, config.bottleneck_size).to(device)
        fake_batch = G(z, src_batch)

    plot_gan_results(src_batch, tgt_batch, fake_batch, chars_to_test, epoch, config, "Specific Test Characters")
    G.train()

