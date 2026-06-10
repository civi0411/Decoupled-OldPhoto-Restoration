const uploadUI = document.getElementById('uploadUI');
const imageUpload = document.getElementById('imageUpload');
const imgBase = document.getElementById('imgBase');
const imgRestored = document.getElementById('imgRestored');
const afterWrapper = document.getElementById('afterWrapper');
const btnStart = document.getElementById('btnStart');
const slider = document.getElementById('slider');
const sliderButton = document.getElementById('sliderButton');

const canvas = document.getElementById('particleCanvas');
const ctx = canvas.getContext('2d');

let animationFrameId;
let particlesArray = [];
let canvasMode = 'none'; // 'uploading', 'processing', 'none'

// Kích hoạt input file khi click vào UI mồi
uploadUI.addEventListener('click', () => { imageUpload.click(); });

// ==========================================
// HỆ THỐNG HẠT (2 CHẾ ĐỘ)
// ==========================================
class Particle {
    constructor(width, height) {
        this.reset(width, height);
    }
    reset(width, height) {
        this.size = Math.random() * 2 + 0.5;
        this.angle = Math.random() * 360;

        if (canvasMode === 'uploading') {
            // Hạt xếp thành các cột thẳng dọc, xuất phát từ dưới
            this.x = Math.random() * width;
            this.baseX = this.x;
            this.y = height + Math.random() * 200; // Nằm dưới viền
            this.speedY = Math.random() * 3 + 2;
        } else if (canvasMode === 'processing') {
            // Hạt phủ kín màn hình
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.baseY = this.y;
            this.density = (Math.random() * 15) + 1;
        }
    }
    update() {
        this.angle += 0.05;

        if (canvasMode === 'uploading') {
            // Hạt bay thẳng lên, lượn sóng nhẹ theo trục X
            this.y -= this.speedY;
            this.x = this.baseX + Math.sin(this.angle) * 10;
            // Nếu bay quá mép trên, reset về dưới
            if (this.y < 0) this.reset(canvas.width, canvas.height);

        } else if (canvasMode === 'processing') {
            // Hạt lượn sóng mạnh theo trục Y, bay ngang X
            this.x += 2;
            this.y = this.baseY + Math.sin(this.angle) * this.density;
            // Nếu bay quá mép phải, vòng lại bên trái
            if (this.x > canvas.width) {
                this.x = 0;
                this.baseY = Math.random() * canvas.height;
            }
        }
    }
    draw() {
        ctx.fillStyle = 'rgba(136, 176, 255, 0.8)';
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fill();
    }
}

function startParticles(mode) {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    canvasMode = mode;
    particlesArray = [];

    // Số lượng hạt tùy mode
    let numberOfParticles = mode === 'processing' ? 400 : 150;

    for (let i = 0; i < numberOfParticles; i++) {
        particlesArray.push(new Particle(canvas.width, canvas.height));
    }

    cancelAnimationFrame(animationFrameId);
    animateParticles();
}

function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (canvasMode !== 'none') {
        for (let i = 0; i < particlesArray.length; i++) {
            particlesArray[i].update();
            particlesArray[i].draw();
        }
        animationFrameId = requestAnimationFrame(animateParticles);
    }
}

// ==========================================
// LUỒNG SỰ KIỆN (UPLOAD -> START -> DONE)
// ==========================================

// 1. Khi chọn ảnh xong
imageUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        // Ẩn chữ Tải ảnh lên
        uploadUI.classList.add('d-none');

        // Bật hiệu ứng hạt bay lên (Uploading effect)
        startParticles('uploading');

        const reader = new FileReader();
        reader.onload = function (event) {
            // Giả lập thời gian load ảnh vào trình duyệt (1.5s)
            setTimeout(() => {
                imgBase.src = event.target.result;
                imgBase.classList.remove('d-none');

                // Tắt hạt upload, hiện nút Start
                canvasMode = 'none';
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                btnStart.classList.remove('d-none');
            }, 1500);
        }
        reader.readAsDataURL(file);
    }
});

// 2. Khi bấm nút START (Gửi API cho model)
btnStart.addEventListener('click', () => {
    // Ẩn nút Start
    btnStart.classList.add('d-none');

    // Blur ảnh gốc
    imgBase.classList.add('is-processing');

    // Bật hiệu ứng hạt lượn sóng phủ kín (Processing effect)
    startParticles('processing');

    // Giả lập thời gian Model xử lý (4 giây)
    setTimeout(() => {
        // Trong thực tế, lúc này có URL trả về từ API
        imgRestored.src = imgBase.src; // Tạm dùng ảnh gốc làm demo

        // Dừng hạt, bỏ blur
        canvasMode = 'none';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        imgBase.classList.remove('is-processing');

        // Hiện thanh Slider và ảnh After
        afterWrapper.classList.remove('d-none');
        slider.classList.remove('d-none');
        sliderButton.classList.remove('d-none');

        // Căn giữa slider
        slider.value = 50;
        afterWrapper.style.width = "50%";
        sliderButton.style.left = "50%";

    }, 4000);
});

// ==========================================
// LOGIC THANH TRƯỢT SO SÁNH
// ==========================================
slider.addEventListener('input', (e) => {
    const val = e.target.value;
    afterWrapper.style.width = val + "%";
    sliderButton.style.left = val + "%";
});