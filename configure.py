def get_default_config(data_name):
    if data_name in ['MNIST-USPS']:
        return dict(
            label=10,
            samples=5000,
            a1=0.95,
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 32],
                arch2=[784, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            training=dict(
                temperature_f=0.5,
                temperature_l=1.0,
                start_dual_prediction=100,
                missing_rate=0.3,
                high_dim=32,
                seed=0,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                lambda1=100,
                lambda2=1,
                lambda3=0.01,
                kernel_mul=2,
                kernel_num=6,
            ),
        )

    elif data_name in ['NoisyMNIST']:
        return dict(
            label=10,
            samples=70000,
            a1=0.95,
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 32],
                arch2=[784, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            training=dict(
                temperature_f=0.5,
                temperature_l=1.0,
                start_dual_prediction=20,
                high_dim=32,
                missing_rate=0.3,
                seed=2,
                batch_size=512,
                epoch=120,
                lr=1.0e-4,
                lambda1=10,
                lambda2=0.001,
                lambda3=0.001,
                kernel_mul=0.02,
                kernel_num=3,
            ),
        )
    else:
        raise Exception('Undefined data_name')
