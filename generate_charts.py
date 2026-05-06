import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_evaluation_charts(csv_path="summary_week9.csv", output_dir="."):
    if not os.path.exists(csv_path):
        print(f"Eroare: Nu am gasit fisierul {csv_path}. Asigura-te ca e in acelasi folder.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ==========================================
    # 1. Bar Chart: Succes (Accepted vs Rejected) in functie de Lumina
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    ax = sns.countplot(
        data=df, 
        x="lighting", 
        hue="decision", 
        palette={"Accepted": "#2ecc71", "Rejected": "#e74c3c"},
        order=["dark", "normal", "bright"] 
    )
    
    plt.title("Rata de Succes a Scanarii în Functie de Iluminare", fontsize=14, fontweight="bold")
    plt.xlabel("Conditii de Iluminare (Lighting)", fontsize=12)
    plt.ylabel("Numar de Imagini", fontsize=12)
    plt.legend(title="Decizie Finala")
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0: 
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    bar_chart_path = os.path.join(output_dir, "success_rate_by_lighting.png")
    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=300) 
    plt.close()
    print(f"Salvat: {bar_chart_path}")

    # ==========================================
    # 2. Pie Chart: Binarizare (Adaptive vs Otsu)
    # ==========================================
    plt.figure(figsize=(7, 7))
    
    method_counts = df['preferred_scan'].value_counts()
    
    colors = ['#3498db', '#f1c40f']
    explode = (0.05, 0) if len(method_counts) > 1 else None 

    plt.pie(
        method_counts, 
        labels=method_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors, 
        explode=explode,
        textprops={'fontsize': 12, 'weight': 'bold'},
        shadow=True
    )
    
    plt.title("Distributia Metodelor de Binarizare Preferate", fontsize=14, fontweight="bold")
    
    pie_chart_path = os.path.join(output_dir, "binarization_pie_chart.png")
    plt.tight_layout()
    plt.savefig(pie_chart_path, dpi=300)
    plt.close()
    print(f"Salvat: {pie_chart_path}")

if __name__ == "__main__":
    generate_evaluation_charts(csv_path="outputs_week9/summary_week9.csv", output_dir=".")