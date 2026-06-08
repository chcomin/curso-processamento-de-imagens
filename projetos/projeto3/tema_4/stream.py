import time
import traceback
import cv2


class VideoStream:
    """Classe para gerenciar um stream de vídeo utilizando OpenCV. Inclui boas práticas de
    checagem de erros e gerenciamento de recursos.
    """
    def __init__(self, cam_id=0, camera_settings=None, window_name="Video Stream"):
        """Inicializa um stream de vídeo.
        
        Args:
            cam_id (int): ID da câmera a ser utilizada.
            camera_settings (dict): Dicionário de configurações da câmera.
            window_name (str): Nome da janela onde o vídeo será exibido
        """

        self.cam_id = cam_id
        self.window_name = window_name
        # No Windows, pode ser necessário passar um segundo argumento cv2.CAP_DSHOW para evitar 
        # problemas com a câmera
        self.vcap = cv2.VideoCapture(self.cam_id)
        if not self.vcap.isOpened():
            raise RuntimeError(f"Erro: Não foi possível acessar a câmera {self.cam_id}")
        
        # Configurações da câmera (opcional)
        if camera_settings:
            for setting, value in camera_settings.items():
                self.vcap.set(setting, value)

    def start(self, processor):
        """Inicia o loop de captura e exibição do vídeo."""

        # Cria uma janela para exibir o vídeo. cv2.WINDOW_AUTOSIZE ajusta o tamanho da janela
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        # Dá uma pequena pausa para garantir que a câmera esteja pronta antes de começar a 
        # capturar frames
        time.sleep(0.5)

        try:
            while True:
                # Obtém próximo frame do vídeo. grabbed indica se o frame foi lido com sucesso, 
                # curr_frame possui a imagem
                grabbed, frame = self.vcap.read()
                if not grabbed:
                    print("Aviso: Não foi possível capturar o frame. Encerrando o stream.")
                    break
                    
                processed_frame = processor.process_frame(frame)
                
                # Exibe a imagem em uma janela nativa do SO
                cv2.imshow(self.window_name, processed_frame)
                
                # cv2.waitKey(x) faz o programa esperar x milisegundos para que uma tecla 
                # seja digitada. Dependendo da plataforma, o resultado dessa função pode 
                # ser um número com mais de 8 bits. O bitwise AND com 0xFF captura apenas
                # os 8 bits menos significativos.
                key = cv2.waitKey(1) & 0xFF
                
                # Se a tecla 'q' for pressionada, encerra o loop
                if key == ord("q"):
                    break
                # Se nenhuma tecla for pressionada, key possuirá o valor 255
                elif key != 255:
                    processor.on_key_press(key)
                    
        except KeyboardInterrupt:
            print("Stream interrompido pelo usuário.")
        except Exception as e:
            print(f"Ocorreu um erro:")
            traceback.print_exc()
        finally:
            # Garante que os recursos sejam liberados mesmo se ocorrer um erro ou interrupção
            self.stop()
    
    def stop(self):
        """Libera os recursos utilizados pelo stream de vídeo."""

        # Libera o dispositivo de captura
        self.vcap.release()
        # Fecha todas as janelas do OpenCV
        cv2.destroyAllWindows()
        # Comando extra para garantir que a janela feche corretamente no MacOS
        for _ in range(5):
            cv2.waitKey(1)

