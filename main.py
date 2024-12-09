from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


def main():
    # Read Video
    video_frames = read_video('input_videos/sample_video.mp4')
    if not video_frames:
        print("Error: No frames were read from the video.")
        return

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # Get object tracks
    try:
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Add object positions to tracks
    tracker.add_position_to_tracks(tracks)

    # Camera Movement Estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    try:
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
            video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl'
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    if 'players' in tracks and tracks['players']:
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    else:
        print("Warning: No players found in tracks.")
        return

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            track['team'] = team
            track['team_color'] = team_assigner.team_colors.get(team, (255, 255, 255))  # Default color if not assigned

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        if frame_num >= len(tracks['ball']):
            print(f"Warning: Missing ball data for frame {frame_num}.")
            team_ball_control.append('No Control')
            continue

        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')
        if not ball_bbox:
            print(f"Warning: No ball bounding box found for frame {frame_num}.")
            team_ball_control.append('No Control')
            continue

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player in player_track:
            player_track[assigned_player]['has_ball'] = True
            team_ball_control.append(player_track[assigned_player].get('team', 'Unknown'))
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 'No Control')

    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')
    print("Output video saved successfully.")

if __name__ == '__main__':
    main()