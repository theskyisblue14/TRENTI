import sys
import numpy
import struct
import time

import pyvisa as visa #import the visa library

DEBUG_MODE = False

class Lecroy():

    def __init__(self):
        print("[*] Lecroy SETUP")
        self.rm = visa.ResourceManager()
        self._scope = None
    
    def __del__(self):
        self.disconnect()
        
    def connect(self,IPAdrress = "10.205.1.95"):
        command = "TCPIP0::" + IPAdrress + "::INSTR"
        print("[0] " + command)
        self._scope = self.rm.open_resource(command)
        self._scope.timeout = 5000
        self._scope.clear()
        self._scope.read_termination = '\n'
        self._scope.write_termination = '\n'
        self._scope.write("COMM_HEADER OFF")
        # self._scope.write(r"""vbs 'app.settodefaultsetup' """)
        command = "*IDN?"
        print("[0] " + command)
        ret = self._scope.query(command)
        self.waitLecroy()
        print("[!] Connected scope:", ret)

    ###
    # Change the volts division of one of the channels
    #
    # @channel
    # String with the name of the channel, can be:
    # "C1", "C2", "C3", "C4",...
    #
    # @return
    # String with the volts division on such channel
    #    
    ###
    def getVoltsDiv(self, channel):
        command = str(channel) + ":VDIV?"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        return self._scope.read()
        
    ###
    # Change the volts division of one of the channels
    #
    # @channel
    # String with the name of the channel, can be:
    # "C1", "C2", "C3", "C4",...
    #
    # @voltsPerDivision
    # Number of Volts per Division, like "1.0" for 1.0 Volts per division or "0.02" for 20 mV of division
    #    
    ###
    def setVoltsDiv(self, channel, voltsPerDivision):
        command = str(channel) + ":" + "VDIV " + voltsPerDivision
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
    
    def getVoltsOffset(self, channel):
        command = str(channel) + ":" + "OFST?"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        return self._scope.read()
        
    def setVoltsOffset(self, channel, voltsOffset):
        command = str(channel) + ":" + "OFST " + voltsOffset
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
    
    def getTimeDiv(self):
        command = "TDIV?"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        return self._scope.read()
        
    def setTimeDiv(self, timeDiv):
        command = "TDIV " + timeDiv
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    def setSampleRate(self, sampleRate):
        command = r"""vbs 'app.acquisition.horizontal.samplerate={}'""".format(sampleRate)
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    ###
    # Get the trigger delay
    #
    # @return
    # String with the amount of delay
    # If the value is negative it will be in seconds format ( After the trigger event )
    # If the value is positive it will be a percentage format ( Before the trigger event )
    #    
    ###                 
    def getTriggerDelay(self):
        command = "TRDL?"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        return self._scope.read()

    ###
    # Set the trigger delay
    #
    # @triggerDelay
    # String with the amount of delay
    # If the value is negative it will be in seconds format ( After the trigger event )
    # If the value is positive it will be a percentage format ( Before the trigger event )
    #    
    ###                 
    def setTriggerDelay(self, triggerDelay):
        command = "TRDL " + triggerDelay
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    ###
    # Get the trigger voltage level
    #
    # @channel
    # String with the name of the channel, can be:
    # "C1", "C2", "C3", "C4", "EX" or "EX10"
    #
    # @return
    # String with the amount of Volts
    #    
    ###          
    def getTriggerLevel(self, channel):
        command = channel + ":" + "TRLV?"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        return self._scope.read()
        
        
    ###
    # Set the trigger voltage level
    #
    # @channel
    # String with the name of the channel, can be:
    # "C1", "C2", "C3", "C4", "EX" or "EX10"
    #
    # @triggerLevel
    # String with the amount of Volts
    #    
    ###                  
    def setTriggerLevel(self, channel, triggerLevel):
        command = channel + ":" + "TRLV " + triggerLevel
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    ###
    # Get the trigger mode
    #
    # @return
    # String with the mode of trigger of choice:
    # "AUTO", "NORM", "SINGLE" or "STOP"
    #
    ###                          
    def getTriggerMode(self):
        command = "TRMD?"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        return self._scope.read()

        
    ###
    # Set the trigger mode
    #
    # @triggerMode
    # String with the mode of trigger of choice:
    # "AUTO", "NORM", "SINGLE" or "STOP"
    #
    ###                          
    def setTriggerMode(self, triggerMode):
        command = "TRMD " + triggerMode
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    ###
    # Get the trigger slope of the specified channel
    #
    # @channel
    # String with the name of the channel, can be:
    # "C1", "C2", "C3", "C4", "EX" or "EX10"
    #
    # @return
    # String with the type of slope wanted
    # "POS", "NEG" or "WINDOW"
    #    
    ###         
    def getTriggerSlope(self, channel):
        command = str(channel) + ":" + "TRSL?"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        return self._scope.read()
        
    ###
    # Set the trigger slope and the channel trigger channel source
    #
    # @channel
    # String with the name of the channel, can be:
    # "C1", "C2", "C3", "C4", "EX" or "EX10"
    #
    # @triggerSlope
    # String with the type of slope wanted
    # "POS", "NEG" or "WINDOW"
    #    
    ###
    def setTriggerSlope(self, channel, triggerSlope):
        command = str(channel) + ":" + "TRSL " + triggerSlope
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
        
    def getPanel(self):
###
###### Code that use the command interface
###
###        panel = self._scope.GetPanel()
        command = "PNSU?"
        if(DEBUG_MODE):
            print("[0] " + command)
        panel = self._scope.query(command)
        # buffer = self._scope.read()
        # while(len(buffer) > 0):
        #     panel += buffer
        #     buffer = self._scope.read()
        return panel
        
    def setPanel(self, panel):
###
###### Code that use the command interface
###
###     self._scope.SetPanel(panel)
       # Transfer temporary file for setup 
       command = r"TRFL DISK,HDD,FILE,'D:\Setups\UserSetup.lss'," + panel
       if(DEBUG_MODE):
           print("[0] " + command)
       self._scope.write(command)
       self._scope.write('')
       # Restore state setup from file
       command = r"RCPN DISK,HDD,FILE,'D:\Setups\UserSetup.lss'"
       if(DEBUG_MODE):
           print("[0] " + command)
       self._scope.write(command)
       # Delete temporary file
       command = r"DELF DISK,HDD,FILE,'D:\Setups\UserSetup.lss'"
       if(DEBUG_MODE):
           print("[0] " + command)
       self._scope.write(command)
    
    def armAndWaitLecroy(self):
        command = "ARM; WAIT"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)

    def stopLecroy(self):
        command = "STOP"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
    
    def enableWaitLecroyAquistion(self):
        command = "WAIT"
        if(DEBUG_MODE):
            print("[0] " + command)
        self._scope.write(command)
    
    def disconnect(self):
        self._scope.close()
        
    def loadLecroyPanelFromFile(self, panelFileName):
        panel = ""
        with open(panelFileName, "rt") as f:
            panel = f.read()
            buffer = f.read()
            if(buffer != ''):
                panel += buffer
                buffer = f.read()
        if(panel != ""):
            self.setPanel(panel)
        
    def storeLecroyPanelToFile(self, panelFileName):
        panel = self.getPanel()
        with open(fileName, "wt") as f:
            f.write(panel)
        
    def getWaveformBinary(self, channel, bytesFormat=False):
        command = "CFMT OFF,"
        if(bytesFormat):
            command += "BYTE,"
        else:
            command += "WORD,"
        command += "BIN"
        if(DEBUG_MODE):
            print("[0]" + command)
        self._scope.write(command)
        # Command that asks for:
        # all points with no SPacing,
        # Number of Points being maximum,
        # First Point is 0,
        # And all Segment Numbers are acquired.
        command = "WFSU SP,0,NP,0,FP,0,SN,0"
        if(DEBUG_MODE):
            print("[0]" + command)
        self._scope.write(command)
        # Bytes are sent with the MSB being the first
        command = "CORD HI"
        if(DEBUG_MODE):
            print("[0]" + command)
        self._scope.write(command)
        command = channel + ":" + "WF?" + " DAT1"
        if(DEBUG_MODE):
            print("[0]" + command)
        self._scope.write(command)
        waveform = self._scope.read_raw(1000)
        buffer = self._scope.read_raw(1000)
        while(len(buffer) > 0):
            waveform += buffer
            buffer = self._scope.read_raw(1000)
        return waveform
        
    def getWaveformDescryption(self, channel, use2BytesDataFormat=True):
        if(use2BytesDataFormat):
            command = 'CFMT ' + 'OFF,'+ 'WORD,'+ 'BIN'
            if(DEBUG_MODE):
                print("[0]" + command)
            self._scope.write(command)        
        else:
            command = 'CFMT ' + 'OFF,'+ 'BYTE,'+ 'BIN'
            if(DEBUG_MODE):
                print("[0]" + command)
            self._scope.write(command)        
        baseCommand = channel + ":" + "INSPECT?"
        command = baseCommand + ' "VERTICAL_GAIN"'
        if(DEBUG_MODE):
            print("[0]" + command)
        answerString = self._scope.query(command)
        answerString = answerString.replace('"', '')
        answerString = answerString.replace(':', '')
        voltageGainString = filter(None, answerString.split(' '))[-1]
        voltageGain = float(voltageGainString)
        command = baseCommand + ' "VERTICAL_OFFSET"'
        if(DEBUG_MODE):
            print("[0]" + command)
        answerString = self._scope.query(command)
        answerString = answerString.replace('"', '')
        answerString = answerString.replace(':', '')
        voltageOffsetString = filter(None, answerString.split(' '))[-1]
        voltageOffset = float(voltageOffsetString)
        command = baseCommand + ' "HORIZ_INTERVAL"'
        if(DEBUG_MODE):
            print("[0]" + command)
        answerString = self._scope.query(command)
        answerString = answerString.replace('"', '')
        answerString = answerString.replace(':', '')
        timeIntervalString = filter(None, answerString.split(' '))[-1]
        timeInterval = float(timeIntervalString)
        command = baseCommand + ' "HORIZ_OFFSET"'
        if(DEBUG_MODE):
            print("[0]" + command)
        answerString = self._scope.query(command)
        answerString = answerString.replace('"', '')
        answerString = answerString.replace(':', '')
        timeOffsetString = filter(None, answerString.split(' '))[-1]
        timeOffset = float(timeOffsetString)
        return voltageGain, voltageOffset, timeInterval, timeOffset
    
    ###
    # Get raw data from lecroy on bytes format.
    #
    # @channel
    # String with the name of the channel, like "C1"
    #
    # @numberOfPoints
    # Number of points that want to be acquired
    #    
    # @firstArray
    # For most cases, for raw channel data it should be 0.
    # In case of dual array waveform, it is possible to put 1 to get from the second array.
    #
    # @use2BytesDataFormat
    # If True, then it will return the y-axis values in 2 bytes format. 
    # If False then it will return the y-axis values in 1 byte format. 
    ###
    def getRawSignal(self, channel, numberOfPoints, firstArray=0, use2BytesDataFormat=True):
        startingPoint = 0           # The point to start transfer (0 = first)
        numberOfPointsToJump = 0    # How many points to jump (0 = get all points, 2 = skip every other point) 
        segmentNumber = 0           # Segment number to get, in case of sequence waveforms.
        # self._scope.SetupWaveformTransfer(startingPoint, numberOfPointsToJump, segmentNumber)
        self._scope.write("WFSU SP," + str(numberOfPointsToJump) + ",NP," + str(numberOfPoints) + ",FP," + str(startingPoint) + ",SN," + str(SegmentNumber))
        command = channel + ":" + "WF?"
        if(use2BytesDataFormat):
            # return self._scope.GetIntegerWaveform(channel, numberOfPoints, firstArray)
            self._scope.write("CFMT OFF,WORD,BIN") ## TODO:TEST, add firstArray parameter
            return self._scope.query_ascii_values(command)
        else:
            # return self._scope.GetByteWaveform(channel, numberOfPoints, firstArray)
            self._scope.write("CFMT OFF,BYTE,BIN")## TODO:TEST
            self._scope.query(command)
    
    ###
    # Get internal pre-processed data from lecroy on 1 byte or 2 bytes format.
    # The difference with this method, is because it has already been preprocessed by lecroy.
    #
    # @channel
    # String with the name of the channel, like "C1"
    #
    # @numberOfPoints
    # Number of points that want to be acquired
    # 
    # @use2BytesDataFormat    
    # If chosen to use 16 bits format it is possible to reload lecroy with the same signal.
    # If chosen the 8 bits format, then it is not possible to load lecroy with the signal.
    #
    # @dataFormat
    # What should be included in the data. Only if value is 5, then it supports being loaded with the same signal.
    # 0 - the descriptor (DESC), 
    # 1 - the user text (TEXT), 
    # 2 - the time descriptor (TIME),
    # 3 - the data (DAT1) block 
    # 4 - a second block of data (DAT2)
    # 5 - all entities (ALL)
    #
    ###
    def getNativeSignalBytes(self, channel, numberOfPoints, use2BytesDataFormat=True, dataFormat=3):
        startingPoint = 0           # The point to start transfer (0 = first)
        numberOfPointsToJump = 0    # How many points to jump (0 = get all points, 2 = skip every other point) 
        SegmentNumber = 0           # Segment number to get, in case of sequence waveforms.
        # self._scope.SetupWaveformTransfer(startingPoint, numberOfPointsToJump, SegmentNumber)
        self._scope.write("WFSU SP," + str(numberOfPointsToJump) + ",NP," + str(numberOfPoints) + ",FP," + str(startingPoint) + ",SN," + str(SegmentNumber))
        if(dataFormat==0):
            internalDataFormat = "DESC"
        elif(dataFormat==1):
            internalDataFormat = "TEXT"
        elif(dataFormat==2):
            internalDataFormat = "TIME"
        elif(dataFormat==3):
            internalDataFormat = "DAT1"
        elif(dataFormat==4):
            internalDataFormat = "DAT2"
        else:
            internalDataFormat = "ALL"
        
        command = channel + ":" + "WF? " + internalDataFormat
        if(use2BytesDataFormat):
            # internalUse2BytesDataFormat = 1
            # receivedBuffer = self._scope.query(command)
            self._scope.write("CFMT OFF,WORD,BIN") ## TODO:TEST, add firstArray parameter
            self._scope.write(command)
            # time.sleep(0.05) #espera para dar tiempo a la traza a ser leido
            # receivedBuffer = self._scope.read_raw()
            # print(len(receivedBuffer))
            receivedBuffer = self._scope.read_bytes(2*numberOfPoints+17)
            # print(len(receivedBuffer))
            # header = receivedBuffer[numberOfPoints*2:]
            # print(header.decode(errors="ignore"))
            receivedBuffer = receivedBuffer[-numberOfPoints*2:]

        else:
            # internalUse2BytesDataFormat = 0
            # receivedBuffer = self._scope.query_binary_values(command, datatype='b', is_big_endian=True)
            self._scope.write("CFMT OFF,BYTE,BIN") ## TODO:TEST, add firstArray parameter
            self._scope.write(command)
            time.sleep(0.05)
            receivedBuffer = self._scope.read_raw()
            receivedBuffer = receivedBuffer[-numberOfPoints:]

        # receivedBuffer = self._scope.GetNativeWaveform(channel, numberOfPoints, internalUse2BytesDataFormat, internalDataFormat)
        if(use2BytesDataFormat):
            interpretedFormat = numpy.frombuffer(receivedBuffer, dtype='>i2')
        else:
            interpretedFormat = numpy.frombuffer(receivedBuffer, dtype='i1')
        return receivedBuffer, interpretedFormat
        
    ###
    # If you got a wave with getNativeSignalBytes, and chose both use2BytesDataFormat=True and dataFormat=5, then 
    # it is possible to send back to lecroy to some specific channels.
    #
    # @channel
    # String with the name of the channel, the ones that work are "M1", "M2", "M3", "M4"
    #
    # @waveform
    # Waveform obtained by getNativeSignalBytes 
    #
    ###
    def setNativeSignalBytes(self, channel, waveform):
        # self._scope.SetNativeWaveform(channel, waveform)
        self._scope.write(str(channel) + ":WF " + waveform)
    
    ### DSO specific functionality (Not sure how to implement with VISA only)
    # ###
    # # Get scaled data from lecroy on float format.
    # #
    # # @channel
    # # String with the name of the channel, like "C1"
    # #
    # # @numberOfPoints
    # # Number of points that want to be acquired
    # #    
    # # @firstArray
    # # For most cases, for raw channel data it should be 0.
    # # In case of dual array waveform, it is possible to put 1 to get from the second array.
    # #
    # # @timeAxis
    # #  Select with the time axis or not
    # ###
    # def getNativeSignalFloat(self, channel, numberOfPoints, firstArray=0, timeAxis=False):
    #     startingPoint = 0           # The point to start transfer (0 = first)
    #     numberOfPointsToJump = 0    # How many points to jump (0 = get all points, 2 = skip every other point) 
    #     segmentNumber = 0           # Segment number to get, in case of sequence waveforms.
    #     if(timeAxis):
    #         interpretedFormat = self._scope.GetScaledWaveformWithTimes(channel, numberOfPoints, firstArray)    
    #         receivedBuffer = [0, 0]
    #         receivedBuffer[0] = struct.pack(str(len(interpretedFormat[0])) + 'f', *interpretedFormat[0])
    #         receivedBuffer[1] = struct.pack(str(len(interpretedFormat[1])) + 'f', *interpretedFormat[1])
    #     else:    
    #         interpretedFormat = self._scope.GetScaledWaveform(channel, numberOfPoints, firstArray)
    #         receivedBuffer = struct.pack(str(len(interpretedFormat)) + 'f', *interpretedFormat)
    #     return receivedBuffer, interpretedFormat
        
    ##
    # Set timeout in seconds
    #
    ##
    def setTransfersTimeout(self, seconds=10):
        self._scope.timeout(seconds)
    
    def waitLecroy(self):
        return self._scope.query("*OPC?")
    ##
    # Resets the entire lecroy
    ##
    def resetLecroy(self):
        # self._scope.DeviceClear(1)
        self._scope.clear()

    ##
    # Resets sweeps
    ##
    def clearSweeps(self):
        # dir(self._scope)
        self._scope.write(r"""VBS 'app.Acquisition.ClearSweeps'""")

    ##
    # Disable display 
    ##
    def displayOff(self):
        cmd = "DISP OFF"
        self._scope.write(cmd)

    ##
    # Enable display 
    ##
    def displayOn(self):
        cmd = "DISP ON"
        self._scope.write(cmd)

    def setTriggerSource(self, channel):
        self._scope.write("TRIG SELECT SNG,SR,"+channel+"")
    
def print_main_class_help():
    print('The parameters options are:')
    print()
    print('lecroy.py -list')
    print("List all scope devices on the network.")
    print()
    print('lecroy.py -r')
    print("Reset Lecroy.")
    print()
    print('lecroy.py -l "LecroyPanel.dat"')
    print("To load Lecroy with a panel.")
    print()
    print('lecroy.py -s "LecroyPanel.dat"')
    print('To store current Lecroy panel into the file "LecroyPanel.dat"')
    print()
    print('lecroy.py -wb "C1" "WaveformByteFormat"')
    print('To store chanel "C1", "C2", "C3" or "C4" y-axis Waveform in Byte format (8 bits) into the file "WaveformByteFormat"')
    print()
    print('lecroy.py -wi "C1" "WaveformIntegerFormat"')
    print('To store chanel "C1", "C2", "C3" or "C4" y-axis Waveform in Integer format (16 bits) into the file "WaveformByteFormat"')
    # print()
    # print('lecroy.py -wf "C1" "WaveformFloatFormat"')
    # print('To store chanel "C1", "C2", "C3" or "C4" y-axis Waveform in Float format (32 bits) into the file "WaveformByteFormat"')


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 4:
        _, waveform, channelName, fileName = sys.argv
        channel_out, channel_out_interpreted = None, None
        if waveform in ['-wb', '-wi', '-wf']:
            print('Trying to store channel: ' + str(channelName) + ' Waveform in file: ' + fileName)
            le = Lecroy()
            le.connect()
            if waveform == "-wb":
                channel_out, channel_out_interpreted = le.getNativeSignalBytes(channelName, 1000000000, False, 3)
            elif waveform == "-wi":
                channel_out, channel_out_interpreted = le.getNativeSignalBytes(channelName, 1000000000, True, 3)
            # elif waveform == "-wf":
            #     channel_out, channel_out_interpreted = le.getNativeSignalFloat(channelName, 1000000000, 0, False)
            else:
                print("Something weird happened")

            numpy.save('interpretedArray.npy', channel_out_interpreted)
            with open(fileName, "wb") as f:
                f.write(channel_out)
            le.disconnect()
        else:
            print('Unknown parameter')
            print()
            print_main_class_help()
    elif argc == 3:
        _, op, fileName = sys.argv
        if op in ['-s', '-l']:
            le = Lecroy()
            le.connect()
            if op == '-s':
                print('Trying to save Panel into file: ' + fileName)
                le.storeLecroyPanelToFile(fileName)
            elif op == '-l':
                print('Trying to load Panel from file: ' + fileName)
                le.loadLecroyPanelFromFile(fileName)
            le.disconnect()
        else:
            print('Unknown parameter')
            print()
            print_main_class_help()
    elif argc == 2:
        if sys.argv[1] == '-r':
            le = Lecroy()
            le.connect()
            le.resetLecroy()
            le.disconnect()
        elif sys.argv[1] == "-list":
            rm = visa.ResourceManager()
            print(rm.list_resources())
        else:
            print('Unknown parameter')
            print()
            print_main_class_help()
    else:
        print("Wrong number of parameters.")
        print()
        print_main_class_help()
