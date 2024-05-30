Read ME

Azure:
1. Connect to host Azure (on terminal ssh -p 5023 [yourid])
2. SU POWERSHELL scp -r -P 5012 "Path/of/your/local/file" user@code/of/your/VM.westeurope.cloudapp.azure.com:/home/disi/
3. BACK TO SSH ls to check what is inside the folder
4. enter in the folder util you reach the file (cd folder)
4. now you can run python main.py
    python main.py --config [config].yaml --run_name [my_run]
