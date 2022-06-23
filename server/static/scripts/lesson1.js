'use strict';

class Lesson1 extends React.Component {
  constructor(props) {
    console.log(props)
    super(props);
    this.state = {
        value:"",
        valueList: [],
        result:""
    };

    this.handleInputChange = this.handleInputChange.bind(this);
  }

  handleInputChange(event) {
    event.preventDefault();   
    let v = event.target.value;  
    this.setState({value: v});
    let inputs = v.split(",")
    let values = []
    let i = 0
    while (i<inputs.length) {
        if (inputs[i].length > 0) {
            values.push(parseInt(inputs[i], 10));
        }
        i++;
    }
    this.state.valueList = values
  }

  getPredict() {
    let data = {
        data: {
            user_values: this.state.valueList
        }
    }
    axios
    .post('/lesson1/lr_predict', data)
    .then(response => {
        var data = response.data.result;
        this.setState({result: data.toString()});
    })
    .catch(function (error) { // 请求失败处理
        console.log(error);
    });

  }

  render() {
    return (
        <div>
            Lesson 1: 简单逻辑回归预测
            <hr/>
            <input type="text" value={this.state.value} onChange={this.handleInputChange}/>
            <br/>
            <button onClick={() => this.getPredict() }>
              获取预测结果
            </button>
            <br/>
            { this.state.result }
        </div>
      );      
 }
}
