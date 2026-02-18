# Configuração do ambiente de desenvolvimento

## Instruções para Linux

Instalar a IDE [VSCode](https://code.visualstudio.com/)

Baixar e instalar o gerenciador de ambientes miniconda neste link:

https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2

No terminal, executar os seguintes comandos:

```bash
# Criação do ambiente chamado cursopdi
conda create -n cursopdi
conda activate cursopdi
conda install -c conda-forge python=3.13 matplotlib numpy scipy pillow opencv

# Nas últimas aulas do curso utilizaremos também redes neurais. Para instalar as bibliotecas associadas:
conda install pytorch torchvision timm
```

## Instruções para Windows

Instalar a IDE [VSCode](https://code.visualstudio.com/)

Você pode instalar seguindo as instruções para windows do link acima, mas recomendo utilizar o Windows Subsystem for Linux (WSL). Vídeo do Youtube com instruções para a instalação do WSL:

https://www.youtube.com/watch?v=0s3pUoqLFeE