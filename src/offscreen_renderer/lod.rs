use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodSettings {
    pub enabled: bool,
    pub distance_check_fps: f32,
    pub high_detail_max_distance: f32,
    pub medium_detail_max_distance: f32,
    pub high_detail_mesh_resolution: usize,
    pub medium_detail_mesh_resolution: usize,
    pub low_detail_mesh_resolution: usize,
}

impl Default for LodSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            distance_check_fps: 12.0,
            high_detail_max_distance: 4.0,
            medium_detail_max_distance: 10.0,
            high_detail_mesh_resolution: 14,
            medium_detail_mesh_resolution: 8,
            low_detail_mesh_resolution: 4,
        }
    }
}

fn resolution_for_distance(distance: f32, lod_settings: LodSettings) -> usize {
    if distance <= lod_settings.high_detail_max_distance {
        lod_settings.high_detail_mesh_resolution
    } else if distance <= lod_settings.medium_detail_max_distance {
        lod_settings.medium_detail_mesh_resolution
    } else {
        lod_settings.low_detail_mesh_resolution
    }
}

struct LodDistanceWorker {
    distance_tx: mpsc::Sender<f32>,
    resolution_rx: mpsc::Receiver<usize>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    settings: Arc<Mutex<LodSettings>>,
}

impl LodDistanceWorker {
    fn new(settings: Arc<Mutex<LodSettings>>) -> Self {
        let (distance_tx, distance_rx) = mpsc::channel::<f32>();
        let (resolution_tx, resolution_rx) = mpsc::channel::<usize>();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_settings = Arc::clone(&settings);

        let handle = thread::spawn(move || {
            let mut latest_distance = None;
            let mut last_resolution = None;

            while !worker_stop.load(Ordering::Relaxed) {
                let interval = {
                    let settings = worker_settings
                        .lock()
                        .ok()
                        .map(|guard| *guard)
                        .unwrap_or_default();
                    let fps = settings.distance_check_fps.max(1.0);
                    Duration::from_secs_f32(1.0 / fps)
                };

                match distance_rx.recv_timeout(interval) {
                    Ok(distance) => {
                        latest_distance = Some(distance);
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }

                while let Ok(distance) = distance_rx.try_recv() {
                    latest_distance = Some(distance);
                }

                let Some(distance) = latest_distance else {
                    continue;
                };

                let settings = worker_settings
                    .lock()
                    .ok()
                    .map(|guard| *guard)
                    .unwrap_or_default();
                if !settings.enabled {
                    continue;
                }

                let resolution = resolution_for_distance(distance, settings);
                if last_resolution != Some(resolution) {
                    let _ = resolution_tx.send(resolution);
                    last_resolution = Some(resolution);
                }
            }
        });

        Self {
            distance_tx,
            resolution_rx,
            stop,
            handle: Some(handle),
            settings,
        }
    }

    fn submit_distance(&self, distance: f32) {
        let _ = self.distance_tx.send(distance);
    }

    fn set_settings(&self, settings: LodSettings) {
        if let Ok(mut guard) = self.settings.lock() {
            *guard = settings;
        }
    }

    fn poll_resolution(&self) -> Option<usize> {
        let mut latest = None;
        while let Ok(resolution) = self.resolution_rx.try_recv() {
            latest = Some(resolution);
        }
        latest
    }
}

impl Drop for LodDistanceWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// LOD設定とバックグラウンドワーカーを一体管理する。
/// 設定の変更はワーカースレッドに自動で伝播する。
pub(super) struct LodManager {
    worker: LodDistanceWorker,
}

impl LodManager {
    pub(super) fn new(settings: LodSettings) -> Self {
        let shared = Arc::new(Mutex::new(settings));
        Self {
            worker: LodDistanceWorker::new(shared),
        }
    }

    pub(super) fn update_settings(&self, settings: LodSettings) {
        self.worker.set_settings(settings);
    }

    pub(super) fn submit_distance(&self, distance: f32) {
        self.worker.submit_distance(distance);
    }

    pub(super) fn poll_resolution(&self) -> Option<usize> {
        self.worker.poll_resolution()
    }
}
