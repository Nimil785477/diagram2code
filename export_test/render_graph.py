import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    fig, ax = plt.subplots()

    ax.add_patch(Rectangle((50, 124), 76, 76, fill=False, linewidth=2))
    ax.text(50+76/2, 124+76/2, 'Node 0', ha='center', va='center')
    ax.add_patch(Rectangle((199, 124), 76, 76, fill=False, linewidth=2))
    ax.text(199+76/2, 124+76/2, 'Node 1', ha='center', va='center')

    # edge 0->1
    na = next(n for n in [{'id': 0, 'bbox': [50, 124, 76, 76]}, {'id': 1, 'bbox': [199, 124, 76, 76]}] if n['id']==0)
    nb = next(n for n in [{'id': 0, 'bbox': [50, 124, 76, 76]}, {'id': 1, 'bbox': [199, 124, 76, 76]}] if n['id']==1)
    ax_a, ay_a, aw_a, ah_a = na['bbox']
    ax_b, ay_b, aw_b, ah_b = nb['bbox']
    x1, y1 = ax_a + aw_a/2, ay_a + ah_a/2
    x2, y2 = ax_b + aw_b/2, ay_b + ah_b/2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=2))

    ax.set_aspect('equal')
    ax.set_xlim(20, 305)
    ax.set_ylim(230, 94)  # invert y to match image coords
    ax.axis('off')
    fig.savefig('outputs/render_graph.png', dpi=200, bbox_inches='tight')
    print('✅ Wrote: outputs/render_graph.png')

if __name__ == '__main__':
    main()
