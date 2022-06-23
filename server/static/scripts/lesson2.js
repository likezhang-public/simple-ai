'use strict';

class MnistButton extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      result: ""
    };
    this.mnistDetect = this.mnistDetect.bind(this);
    this.restart = this.restart.bind(this);
  }

  restart() {
    var canvas = document.getElementById("lesson2_canvas");
    var canvasCtx = canvas.getContext("2d");
    canvasCtx.fillStyle = "white";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);    
    canvasCtx.fillStyle = "#000000";
  }

  mnistDetect() {
    var canvas = document.getElementById("lesson2_canvas");
    const b64data = canvas.toDataURL('image/jpeg');
    console.log(b64data)

    let postData = {
      data: b64data
    }
    axios
    .post('/lesson2/mnist_predict', postData)
    .then(response => {
        var result = response.data.result;
        console.log(result);
        this.setState({result: "识别结果: " + result})
        
    })
    .catch(function (error) { // 请求失败处理
        console.log(error);
    });

  }

  render() {
    return (
      <div>
        <button onClick={() => this.mnistDetect() }>
          识别所绘数字
        </button>
        <br/>
        { this.state.result }
        <br/>
        <button onClick={() => this.restart() }>
          再来一次
        </button>
      </div>
      );      
 }
}

class Lesson2Canvas extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      startDrawing: false,
    }

    this.drawing = this.drawing.bind(this);
    this.start = this.start.bind(this);
    this.stop = this.stop.bind(this);
  }

  ctx = null;

  componentDidMount() {
    var canvas = document.getElementById("lesson2_canvas");
    var canvasCtx = canvas.getContext("2d");
    canvasCtx.fillStyle = "white";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);    
    canvasCtx.fillStyle = "#000000";
    this.ctx = canvasCtx
  }

  drawing(e) {
    if (this.state.startDrawing == false) {
      return
    }

    let x = e.nativeEvent.offsetX
    let y = e.nativeEvent.offsetY
    var size = 12;
    this.ctx.fillRect(x-size/2, y-size/2, size, size);
  }

  start() {
    this.setState({startDrawing: true});
  }

  stop() {
    this.setState({startDrawing: false});
  }

  render() {
    return (
        <canvas
          id={"lesson2_canvas"}
          width={150}
          height={150}
          style={{
            position: "relative",
            left: 200,
            top: 100,
            border: "1px solid #c3c3c3"
          }}
          onMouseUp={this.stop}
          onMouseDown={this.start}
          onMouseMove={(e)=>{this.drawing(e)}}
        >
          Your browser does not support the canvas element.
        </canvas>
      );      
 }
}


class Lesson2 extends React.Component {
  constructor(props) {
    console.log(props)
    super(props);
    this.state = {
        result:""
    };

    this.handleCommitChange = this.handleCommitChange.bind(this);
  }

  handleCommitChange(event) {
    event.preventDefault();
  }


  render() {
    return (
        <div>
            Lesson 2: pytorch mnist 手写数字识别
            <hr/>
            在下面方框中绘制任意一个0~9之间的数字（必须是单个数字）
            <div>
            <Lesson2Canvas />
            </div>
            <div>
            <MnistButton />
            </div>
        </div>
      );      
 }
}
