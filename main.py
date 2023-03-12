from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
import argparse

BoardShim.enable_dev_board_logger()

# configure argparse
parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
args = parser.parse_args()

params = BrainFlowInputParams()
params.serial_port = args.serial_port


board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
board.prepare_session()
board.start_stream ()
time.sleep(10)
# data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
data = board.get_board_data()  # get all data and remove it from internal buffer
board.stop_stream()
board.release_session()

print(data)