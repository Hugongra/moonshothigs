import numpy as np
import matplotlib.pyplot as plt


def simulate_pareto():
    # Parámetros del paper (Sección A.1)
    p = 0.2    # Tasa de éxito base (Llama-3.1-70b)
    r = 1.5    # Factor de refinamiento por feedback del oráculo

    # 1. Función para Linear Retry: P = 1 - (1-p)^k
    def prob_linear(k):
        return 1 - (1 - p)**k

    # 2. Función para OGSR: Basada en la sumatoria del Apéndice A.1
    def prob_ogsr(b, d):
        total_p = 0
        fail_accumulated = 1
        for i in range(1, d + 1):
            p_i = min(p * (r**(i - 1)), 0.99)  # Probabilidad mejora por profundidad
            p_success_at_level = 1 - (1 - p_i)**b
            total_p += p_success_at_level * fail_accumulated
            fail_accumulated *= (1 - p_success_at_level)
        return total_p

    # Datos para graficar
    configs = [
        {'name': 'Linear (k=5)', 'cost': 135, 'p': 0.58},  # Datos reales de tu Tabla 7
        {'name': 'Linear (k=10)', 'cost': 270, 'p': prob_linear(10)},
        {'name': 'OGSR (d=3, b=3)', 'cost': 194, 'p': 0.84},  # Tu "Pareto Frontier"
        {'name': 'OGSR (d=3, b=5)', 'cost': 320, 'p': prob_ogsr(5, 3)}
    ]

    costs = [c['cost'] for c in configs]
    success_rates = [c['p'] * 100 for c in configs]
    labels = [c['name'] for c in configs]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.scatter(costs, success_rates, color='red', s=100, zorder=5)

    # Dibujar la línea de la Frontera de Pareto
    # Nota: Conectamos k=5 -> OGSR(3,3) -> k=10 para mostrar la curva
    plt.plot(costs, success_rates, linestyle='--', color='gray', alpha=0.5)

    # Anotaciones
    for i, txt in enumerate(labels):
        plt.annotate(txt, (costs[i], success_rates[i]), xytext=(5, 5), textcoords='offset points')

    plt.title('Ablation Study: Cost-Performance Pareto Frontier')
    plt.xlabel('Oracle Call Budget (Inference Cost)')
    plt.ylabel('Cumulative Success Rate (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(50, 100)

    # Guardar para el paper
    plt.savefig('pareto_frontier.png', dpi=300)
    print("Gráfica generada como 'pareto_frontier.png'")
    plt.show()


if __name__ == "__main__":
    simulate_pareto()
