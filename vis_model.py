from phd_lab.models import resnet18_ri_hybrid2

if __name__ == '__main__':
    model = resnet18_ri_hybrid2()
    from rfa_toolbox import create_graph_from_pytorch_model, visualize_architecture
    print(model)
    g = create_graph_from_pytorch_model(model)
    visualize_architecture(g, "resnet").view()