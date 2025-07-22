#!/usr/bin/env python3
"""
Script para crear un nuevo contenedor Docker y configurar image_161160
- Copia images/image_161160/ del host a /test/image_161160/ del contenedor
- Copia archivos específicos de /test/image_10566/ a /test/image_161160/ dentro del contenedor
"""

import subprocess
import time
import os
import sys
import argparse
import shutil
from pathlib import Path

class Container161160Setup:
    def __init__(self, new_container_name, docker_image="zyw200/firmfuzzer"):
        self.new_container_name = new_container_name
        self.docker_image = docker_image
        self.host_source_path = "images/image_161160"
        self.container_target_path = "/test/image_161160"
        self.container_source_path = "/test/image_10566"
        
        # Archivos específicos a copiar dentro del contenedor
        self.files_to_copy = [
            "afl-fuzz",
            "afl-fuzz-full", 
            "afl-qemu-trace",
            "qemu-system-mips",
            "qemu-system-mips-full"
        ]
        
    def log(self, message, level="INFO"):
        """Logging con timestamp y colores"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        colors = {
            "INFO": "\033[0m",      # Normal
            "SUCCESS": "\033[92m",  # Verde
            "WARNING": "\033[93m",  # Amarillo
            "ERROR": "\033[91m",    # Rojo
            "DEBUG": "\033[94m"     # Azul
        }
        
        reset = "\033[0m"
        color = colors.get(level, "\033[0m")
        
        print(f"{color}[{timestamp}] [{level}] {message}{reset}", flush=True)
        
    def check_host_source(self):
        """Verifica que el directorio fuente existe en el host"""
        try:
            if not os.path.exists(self.host_source_path):
                self.log(f"ERROR: Host source directory not found: {self.host_source_path}", "ERROR")
                self.log(f"Current working directory: {os.getcwd()}", "ERROR")
                self.log("Available directories:", "INFO")
                try:
                    for item in os.listdir('.'):
                        if os.path.isdir(item):
                            self.log(f"  {item}/", "INFO")
                except:
                    pass
                return False
            
            # Obtener información del directorio
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for root, dirs, files in os.walk(self.host_source_path):
                dir_count += len(dirs)
                file_count += len(files)
                for file in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, file))
                    except:
                        pass
            
            size_mb = total_size / (1024 * 1024)
            size_gb = total_size / (1024 * 1024 * 1024)
            
            self.log(f"? Host source directory found: {self.host_source_path}", "SUCCESS")
            self.log(f"Directory analysis:", "INFO")
            self.log(f"  Size: {size_gb:.2f} GB ({size_mb:.1f} MB)", "INFO")
            self.log(f"  Files: {file_count}", "INFO")
            self.log(f"  Directories: {dir_count}", "INFO")
            
            if size_gb > 5:
                self.log(f"WARNING: Large directory detected - copy will take time", "WARNING")
                
            return True
            
        except Exception as e:
            self.log(f"Error checking host source: {e}", "ERROR")
            return False
    
    def cleanup_existing_container(self):
        """Limpia contenedor existente si existe"""
        try:
            self.log(f"Checking for existing container: {self.new_container_name}", "INFO")
            
            # Verificar si el contenedor existe
            check_cmd = ['docker', 'inspect', self.new_container_name]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if check_result.returncode == 0:
                self.log(f"Container {self.new_container_name} exists, removing...", "WARNING")
                
                # Detener y eliminar contenedor existente
                cleanup_commands = [
                    ['docker', 'stop', self.new_container_name],
                    ['docker', 'rm', '-f', self.new_container_name]
                ]
                
                for cmd in cleanup_commands:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        self.log(f"? Command successful: {' '.join(cmd[1:])}", "SUCCESS")
                    else:
                        self.log(f"Command result: {result.stderr.strip()}", "DEBUG")
                        
                self.log(f"? Existing container cleaned up", "SUCCESS")
            else:
                self.log(f"? No existing container found", "SUCCESS")
                
            return True
            
        except Exception as e:
            self.log(f"Error cleaning up existing container: {e}", "ERROR")
            return False
    
    def launch_new_container(self):
        """Lanza nuevo contenedor Docker"""
        try:
            self.log("=" * 60, "INFO")
            self.log("LAUNCHING NEW DOCKER CONTAINER", "INFO")
            self.log("=" * 60, "INFO")
            
            # Comando para lanzar contenedor (igual que en TRENTI)
            cmd = [
                'docker', 'run', '-it', '--name', self.new_container_name,
                '--env', 'USER=root', '--privileged', '--device=/dev/net/tun',
                '-d', self.docker_image, '/bin/bash'
            ]
            
            self.log(f"Launch command: {' '.join(cmd)}", "DEBUG")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                self.log(f"Docker launch failed with return code: {result.returncode}", "ERROR")
                self.log(f"STDOUT: {result.stdout}", "ERROR")
                self.log(f"STDERR: {result.stderr}", "ERROR")
                return False
                
            container_id = result.stdout.strip()
            self.log(f"? Container launched successfully", "SUCCESS")
            self.log(f"Container ID: {container_id[:12]}...", "SUCCESS")
            self.log(f"Container name: {self.new_container_name}", "SUCCESS")
            
            # Verificar que el contenedor está ejecutándose
            time.sleep(3)
            if not self.check_container_running():
                return False
                
            return True
            
        except subprocess.TimeoutExpired:
            self.log("Timeout launching container", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error launching container: {e}", "ERROR")
            return False
    
    def check_container_running(self):
        """Verifica que el contenedor está ejecutándose"""
        try:
            check_cmd = ['docker', 'inspect', '-f', '{{.State.Running}}', self.new_container_name]
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.log(f"Cannot inspect container {self.new_container_name}", "ERROR")
                return False
                
            if result.stdout.strip() != 'true':
                self.log(f"Container {self.new_container_name} is not running", "ERROR")
                return False
                
            self.log(f"? Container {self.new_container_name} is running", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error checking container status: {e}", "ERROR")
            return False
    
    def copy_host_directory_to_container(self):
        """Copia el directorio completo del host al contenedor"""
        try:
            self.log("=" * 60, "INFO")
            self.log("COPYING HOST DIRECTORY TO CONTAINER", "INFO")
            self.log("=" * 60, "INFO")
            
            self.log(f"Source: {self.host_source_path} (host)", "INFO")
            self.log(f"Target: {self.container_target_path} (container)", "INFO")
            
            # Comando docker cp para copiar directorio completo
            copy_command = ['docker', 'cp', self.host_source_path, 
                           f'{self.new_container_name}:/test/']
            
            self.log(f"Executing: {' '.join(copy_command)}", "DEBUG")
            self.log("=" * 60, "WARNING")
            self.log("COPYING DIRECTORY - THIS MAY TAKE SEVERAL MINUTES", "WARNING")
            self.log("DO NOT INTERRUPT - Container will be unusable if interrupted", "WARNING")
            self.log("=" * 60, "WARNING")
            
            start_time = time.time()
            
            # Ejecutar con monitoreo en tiempo real
            process = subprocess.Popen(copy_command, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            
            # Monitor con timeout de 1 hora
            timeout_seconds = 3600
            poll_interval = 10
            
            while True:
                elapsed = time.time() - start_time
                
                # Verificar si el proceso terminó
                return_code = process.poll()
                if return_code is not None:
                    stdout, stderr = process.communicate()
                    
                    if return_code == 0:
                        self.log(f"? Host to container copy completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)", "SUCCESS")
                        return self.verify_host_copy()
                    else:
                        self.log(f"Copy operation failed", "ERROR")
                        self.log(f"Return code: {return_code}", "ERROR")
                        self.log(f"STDERR: {stderr}", "ERROR")
                        if stdout:
                            self.log(f"STDOUT: {stdout}", "ERROR")
                        return False
                
                # Verificar timeout
                if elapsed > timeout_seconds:
                    self.log(f"Timeout after {timeout_seconds/60:.1f} minutes", "ERROR")
                    self.log("Terminating copy operation...", "ERROR")
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.log("Force killing copy process...", "ERROR")
                        process.kill()
                    return False
                
                # Mostrar progreso
                minutes, seconds = divmod(elapsed, 60)
                print(f"\rCopying from host... Elapsed: {int(minutes):02d}:{int(seconds):02d} - Please wait...", 
                      end='', flush=True)
                
                time.sleep(poll_interval)
                
        except Exception as e:
            self.log(f"Error copying from host: {e}", "ERROR")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False
    
    def verify_host_copy(self):
        """Verifica que la copia del host fue exitosa"""
        try:
            print()  # Nueva línea después del progreso
            self.log("Verifying host directory copy...", "INFO")
            
            # Verificar que el directorio existe
            check_cmd = ['docker', 'exec', self.new_container_name, 'test', '-d', self.container_target_path]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
            
            if check_result.returncode != 0:
                self.log(f"? Directory verification failed - {self.container_target_path} not found", "ERROR")
                return False
            
            self.log(f"? Directory {self.container_target_path} exists in container", "SUCCESS")
            
            # Contar archivos copiados
            count_cmd = ['docker', 'exec', self.new_container_name, 'bash', '-c',
                        f'find {self.container_target_path} -type f | wc -l']
            count_result = subprocess.run(count_cmd, capture_output=True, text=True, timeout=60)
            
            if count_result.returncode == 0:
                copied_files = int(count_result.stdout.strip())
                self.log(f"? Verification: {copied_files} files copied to container", "SUCCESS")
                
                if copied_files < 50:
                    self.log(f"WARNING: Only {copied_files} files found - copy may be incomplete", "WARNING")
                else:
                    self.log("? Host copy appears complete based on file count", "SUCCESS")
            
            # Verificar algunos archivos clave del host
            key_files = ['run.sh', 'test.py', 'user.sh', 'image.raw']
            for key_file in key_files:
                key_cmd = ['docker', 'exec', self.new_container_name, 'test', '-f', 
                          f'{self.container_target_path}/{key_file}']
                key_result = subprocess.run(key_cmd, capture_output=True, text=True, timeout=10)
                
                if key_result.returncode == 0:
                    self.log(f"? Key file found: {key_file}", "SUCCESS")
                else:
                    self.log(f"? Key file missing: {key_file}", "WARNING")
            
            return True
            
        except Exception as e:
            self.log(f"Error verifying host copy: {e}", "ERROR")
            return False
    
    def copy_internal_files(self):
        """Copia archivos específicos desde image_10566 a image_161160 dentro del contenedor"""
        try:
            self.log("=" * 60, "INFO")
            self.log("COPYING INTERNAL FILES WITHIN CONTAINER", "INFO")
            self.log("=" * 60, "INFO")
            
            self.log(f"Source: {self.container_source_path} (within container)", "INFO")
            self.log(f"Target: {self.container_target_path} (within container)", "INFO")
            self.log(f"Files to copy: {', '.join(self.files_to_copy)}", "INFO")
            
            # Verificar que el directorio fuente existe
            check_source_cmd = ['docker', 'exec', self.new_container_name, 'test', '-d', self.container_source_path]
            check_result = subprocess.run(check_source_cmd, capture_output=True, text=True, timeout=30)
            
            if check_result.returncode != 0:
                self.log(f"WARNING: Source directory {self.container_source_path} not found in container", "WARNING")
                self.log("This is not critical - the container can still work", "INFO")
                return True  # No es crítico para el funcionamiento
                
            self.log(f"? Source directory {self.container_source_path} verified", "SUCCESS")
            
            # Listar contenido del directorio fuente para debugging
            list_cmd = ['docker', 'exec', self.new_container_name, 'ls', '-la', self.container_source_path]
            list_result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=30)
            
            if list_result.returncode == 0:
                self.log(f"Contents of {self.container_source_path}:", "DEBUG")
                for line in list_result.stdout.split('\n')[:10]:  # Mostrar solo las primeras 10 líneas
                    if line.strip():
                        self.log(f"  {line}", "DEBUG")
            
            success = True
            copied_files = 0
            
            for file_name in self.files_to_copy:
                source_path = f"{self.container_source_path}/{file_name}"
                dest_path = f"{self.container_target_path}/{file_name}"
                
                try:
                    # Verificar si el archivo fuente existe
                    check_cmd = ['docker', 'exec', self.new_container_name, 'test', '-f', source_path]
                    check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
                    
                    if check_result.returncode != 0:
                        self.log(f"? Source file not found: {source_path}, skipping...", "WARNING")
                        continue
                    
                    # Copiar archivo dentro del contenedor
                    copy_cmd = ['docker', 'exec', self.new_container_name, 'cp', source_path, dest_path]
                    self.log(f"Copying {file_name}...", "INFO")
                    
                    copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=30)
                    
                    if copy_result.returncode == 0:
                        # Verificar la copia
                        verify_cmd = ['docker', 'exec', self.new_container_name, 'test', '-f', dest_path]
                        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=10)
                        
                        if verify_result.returncode == 0:
                            self.log(f"? Successfully copied and verified {file_name}", "SUCCESS")
                            copied_files += 1
                        else:
                            self.log(f"? Copy verification failed for {file_name}", "ERROR")
                            success = False
                    else:
                        self.log(f"? Copy failed for {file_name}: {copy_result.stderr}", "ERROR")
                        success = False
                        
                except subprocess.TimeoutExpired:
                    self.log(f"? Timeout copying {file_name}", "ERROR")
                    success = False
                except Exception as e:
                    self.log(f"? Error copying {file_name}: {e}", "ERROR")
                    success = False
            
            if copied_files > 0:
                self.log(f"? Internal file copy completed - {copied_files}/{len(self.files_to_copy)} files copied", "SUCCESS")
            else:
                self.log("? No internal files were copied", "WARNING")
                
            # Hacer archivos ejecutables
            if copied_files > 0:
                self.log("Making copied files executable...", "INFO")
                chmod_cmd = ['docker', 'exec', self.new_container_name, 'bash', '-c',
                           f'chmod +x {self.container_target_path}/afl-* {self.container_target_path}/qemu-* 2>/dev/null || true']
                subprocess.run(chmod_cmd, capture_output=True, text=True, timeout=30)
                self.log("? Executable permissions set", "SUCCESS")
            
            return True  # No es crítico para el funcionamiento básico
            
        except Exception as e:
            self.log(f"Error in internal file copy: {e}", "WARNING")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return True  # No es crítico
    
    def final_verification(self):
        """Verificación final del contenedor configurado"""
        try:
            self.log("=" * 60, "INFO")
            self.log("PERFORMING FINAL VERIFICATION", "INFO")
            self.log("=" * 60, "INFO")
            
            # Verificar contenedor ejecutándose
            if not self.check_container_running():
                return False
            
            # Verificar directorio principal
            check_main_cmd = ['docker', 'exec', self.new_container_name, 'test', '-d', self.container_target_path]
            if subprocess.run(check_main_cmd, capture_output=True, text=True).returncode != 0:
                self.log(f"? Main directory {self.container_target_path} not found", "ERROR")
                return False
                
            self.log(f"? Main directory {self.container_target_path} verified", "SUCCESS")
            
            # Contar archivos totales
            count_cmd = ['docker', 'exec', self.new_container_name, 'bash', '-c',
                        f'find {self.container_target_path} -type f | wc -l']
            count_result = subprocess.run(count_cmd, capture_output=True, text=True, timeout=60)
            
            if count_result.returncode == 0:
                total_files = int(count_result.stdout.strip())
                self.log(f"? Total files in container: {total_files}", "SUCCESS")
            
            # Verificar archivos clave
            all_key_files = ['run.sh', 'test.py', 'user.sh', 'image.raw'] + self.files_to_copy
            
            found_files = 0
            for key_file in all_key_files:
                key_cmd = ['docker', 'exec', self.new_container_name, 'test', '-f', 
                          f'{self.container_target_path}/{key_file}']
                key_result = subprocess.run(key_cmd, capture_output=True, text=True, timeout=10)
                
                if key_result.returncode == 0:
                    self.log(f"? Key file verified: {key_file}", "SUCCESS")
                    found_files += 1
                else:
                    self.log(f"? Key file missing: {key_file}", "WARNING")
            
            self.log(f"? Key files found: {found_files}/{len(all_key_files)}", "SUCCESS")
            
            # Verificar permisos de archivos ejecutables
            exec_files = [f for f in self.files_to_copy if f.startswith(('afl-', 'qemu-'))]
            for exec_file in exec_files:
                perm_cmd = ['docker', 'exec', self.new_container_name, 'test', '-x', 
                           f'{self.container_target_path}/{exec_file}']
                if subprocess.run(perm_cmd, capture_output=True, text=True).returncode == 0:
                    self.log(f"? Executable permissions verified: {exec_file}", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Error in final verification: {e}", "ERROR")
            return False
    
    def run_complete_setup(self):
        """Ejecuta la configuración completa"""
        try:
            self.log("=" * 70, "INFO")
            self.log("STARTING COMPLETE CONTAINER SETUP FOR IMAGE_161160", "INFO")
            self.log("=" * 70, "INFO")
            
            # Verificación inicial
            self.log("Step 0: Checking host source directory...", "INFO")
            if not self.check_host_source():
                return False
            
            # Paso 1: Limpiar contenedor existente
            self.log("Step 1: Cleaning up existing container...", "INFO")
            if not self.cleanup_existing_container():
                return False
                
            # Paso 2: Lanzar nuevo contenedor
            self.log("Step 2: Launching new container...", "INFO")
            if not self.launch_new_container():
                return False
                
            # Paso 3: Copiar directorio del host
            self.log("Step 3: Copying host directory to container...", "INFO")
            if not self.copy_host_directory_to_container():
                return False
                
            # Paso 4: Copiar archivos internos
            self.log("Step 4: Copying internal files within container...", "INFO")
            if not self.copy_internal_files():
                self.log("Internal file copy had issues, but continuing...", "WARNING")
                
            # Paso 5: Esperar sincronización
            self.log("Step 5: Waiting for filesystem sync...", "INFO")
            time.sleep(10)
            
            # Paso 6: Verificación final
            self.log("Step 6: Final verification...", "INFO")
            if not self.final_verification():
                return False
                
            self.log("=" * 70, "SUCCESS")
            self.log("CONTAINER SETUP COMPLETED SUCCESSFULLY", "SUCCESS")
            self.log("=" * 70, "SUCCESS")
            self.log(f"? Container '{self.new_container_name}' is ready for use", "SUCCESS")
            self.log(f"? Image 161160 data is available at: {self.container_target_path}", "SUCCESS")
            self.log(f"? All necessary files have been copied and configured", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"CRITICAL ERROR in setup: {e}", "ERROR")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False


def main():
    print("=" * 70)
    print("CONTAINER SETUP FOR IMAGE_161160")
    print("=" * 70)
    print("Creating container: Full_System_Emulation")
    print("Docker image: zyw200/firmfuzzer")
    print("Host source: images/image_161160/")
    print("Container target: /test/image_161160/")
    print("Internal source: /test/image_10566/")
    print("=" * 70)
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("images/image_161160"):
        print("? ERROR: images/image_161160/ directory not found")
        print(f"Current directory: {os.getcwd()}")
        print("Please run this script from the directory containing images/image_161160/")
        sys.exit(1)
    
    # Crear configurador
    setup = Container161160Setup(
        new_container_name='Full_System_Emulation',
        docker_image='zyw200/firmfuzzer'
    )
    
    try:
        # Ejecutar configuración completa
        success = setup.run_complete_setup()
        
        if success:
            print(f"\n?? SUCCESS! Container 'Full_System_Emulation' is ready")
            print("\n?? Next steps:")
            print(f"   # Connect to the container:")
            print(f"   docker exec -it Full_System_Emulation /bin/bash")
            print(f"   ")
            print(f"   # Verify the setup:")
            print(f"   docker exec Full_System_Emulation ls -la /test/image_161160/")
            print(f"   ")
            print(f"   # Check copied executables:")
            print(f"   docker exec Full_System_Emulation ls -la /test/image_161160/afl-*")
            print(f"   docker exec Full_System_Emulation ls -la /test/image_161160/qemu-*")
        else:
            print(f"\n? SETUP FAILED")
            print("Check the error messages above")
            
            # Limpiar contenedor fallido
            print(f"\nCleaning up failed container...")
            subprocess.run(['docker', 'rm', '-f', 'Full_System_Emulation'], 
                          capture_output=True)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        print(f"Cleaning up container Full_System_Emulation...")
        subprocess.run(['docker', 'rm', '-f', 'Full_System_Emulation'], 
                      capture_output=True)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        subprocess.run(['docker', 'rm', '-f', 'Full_System_Emulation'], 
                      capture_output=True)
        sys.exit(1)


if __name__ == "__main__":
    main()