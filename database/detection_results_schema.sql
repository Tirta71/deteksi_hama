CREATE TABLE IF NOT EXISTS detection_results (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  filename VARCHAR(255) NOT NULL,

  full_result_image VARCHAR(255),
  full_classes TEXT,
  full_total_pests INT DEFAULT 0,
  full_avg_conf DECIMAL(8, 4) DEFAULT 0,
  full_duration_ms INT DEFAULT 0,

  crop_result_image VARCHAR(255),
  crop_classes TEXT,
  crop_total_pests INT DEFAULT 0,
  crop_avg_conf DECIMAL(8, 4) DEFAULT 0,
  crop_duration_ms INT DEFAULT 0,
  per_grid_results_json LONGTEXT,

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT fk_detection_results_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE SET NULL
);
